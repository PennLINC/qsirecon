# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loading atlases
^^^^^^^^^^^^^^^

"""

import logging

LOGGER = logging.getLogger('nipype.interface')

#: The template atlases are taken from when nothing says which one to use
DEFAULT_ATLAS_TEMPLATE = 'MNI152NLin2009cAsym'


def collect_atlases(datasets, atlases, bids_filters={}, xfm_edges=None):
    """Collect atlases from a list of BIDS-Atlas datasets.

    Selection of labels files and metadata does not leverage the inheritance principle.
    That probably won't be possible until PyBIDS supports the BIDS-Atlas extension natively.

    An atlas may be available in more than one template.
    When ``xfm_edges`` is given, the copy with the shortest chain of transforms to the
    anatomical space is used, whatever the output space is: atlases are warped into the
    space of the DWI, so the template they come from does not show in the outputs.
    A template the input dataset was normalized to is one transform away.
    Any other template is reached through TemplateFlow's template-to-template transforms,
    which are downloaded, and a warning names the chain.
    Between equally long chains, a copy in the same template as the one the input dataset
    was normalized to (in a different cohort) is preferred.

    Parameters
    ----------
    datasets : dict of str:str or str:BIDSLayout pairs
        Dictionary of BIDS datasets to search for atlases.
    atlases : list of str
        List of atlases to collect from across the datasets.
    bids_filters : dict
        Additional filters to apply to the BIDS query.
        Only the "atlas" key is used.
        A "template" entry restricts the templates atlases may be taken from.
    xfm_edges : list of :class:`~qsirecon.utils.transforms.Xfm`, optional
        The template transforms in the input dataset.
        If None, no transforms are looked for and, unless ``bids_filters`` names a
        template, atlases are taken from MNI152NLin2009cAsym.

    Returns
    -------
    atlas_cache : dict
        Dictionary of atlases with metadata.
        Keys are the atlas names, values are dictionaries with keys:

        - "dataset" : str
            Name of the dataset containing the atlas.
        - "image" : str
            Path to the atlas image.
        - "labels" : str
            Path to the atlas labels file.
        - "metadata" : dict
            Metadata associated with the atlas.
        - "space" : str
            The template the atlas image is in, with any cohort appended as "+<cohort>".

        and, when ``xfm_edges`` is given:

        - "xfm_to_anat" : list of str
            The transforms that bring the atlas into the anatomical space,
            in the order ANTs takes them.
        - "xfm_path" : list of str
            The spaces those transforms go through, from the atlas's to the anatomical one.
    """
    import json

    import pandas as pd
    from bids.layout import BIDSLayout

    from qsirecon.data import load as load_data

    atlas_cfg = load_data('atlas_bids_config.json')
    bids_filters = bids_filters or {}

    atlas_filter = dict(bids_filters.get('atlas', {}))
    atlas_filter['suffix'] = atlas_filter.get('suffix') or 'dseg'  # XCP-D only supports dsegs
    atlas_filter['extension'] = ['.nii.gz', '.nii']
    if xfm_edges is None and not atlas_filter.get('template'):
        atlas_filter['template'] = DEFAULT_ATLAS_TEMPLATE
    elif not atlas_filter.get('template'):
        # Any template will do: the one with the shortest chain of transforms is chosen
        atlas_filter.pop('template', None)

    layouts = {}
    # The copies of each atlas: {atlas: {space: {'dataset': ..., 'image': ...}}}
    candidates = {atlas: {} for atlas in atlases}
    errors = []
    for dataset_name, dataset_path in datasets.items():
        LOGGER.info(f'Collecting atlases from {dataset_name}: {dataset_path}')
        if not isinstance(dataset_path, BIDSLayout):
            layout = BIDSLayout(
                dataset_path,
                config=['bids', 'derivatives', atlas_cfg],
                validate=False,
            )
        else:
            layout = dataset_path

        if layout.get_dataset_description().get('DatasetType') != 'derivative':
            continue

        layouts[dataset_name] = layout
        for atlas in atlases:
            atlas_files = layout.get(atlas=atlas, **atlas_filter)
            by_space = {}
            for atlas_file in atlas_files:
                by_space.setdefault(_atlas_space(atlas_file), []).append(atlas_file.path)

            for space, images in by_space.items():
                if len(images) > 1:
                    bulleted_list = '\n'.join([f'  - {img}' for img in images])
                    LOGGER.warning(
                        f'Multiple atlas images found for {atlas} in {space} with query '
                        f'{atlas_filter}:\n{bulleted_list}\nUsing {images[0]}.'
                    )

                if space in candidates[atlas]:
                    raise ValueError(f"Multiple datasets contain the same atlas '{atlas}'")

                candidates[atlas][space] = {'dataset': dataset_name, 'image': images[0]}

    atlas_cache = {}
    for atlas in atlases:
        if not candidates[atlas]:
            LOGGER.warning(f'No atlas images found for {atlas} with query {atlas_filter}')
            errors.append(f'No atlas images found for {atlas} with query {atlas_filter}')
            continue

        spaces = sorted(candidates[atlas])
        chain = {}
        if xfm_edges is None:
            space = spaces[0]
            if len(spaces) > 1:
                LOGGER.warning(f'{atlas} is available in {", ".join(spaces)}. Using {space}.')
        else:
            try:
                chain = _chain_atlas_to_anat(atlas, spaces, xfm_edges)
            except ValueError as exc:
                errors.append(str(exc))
                continue

            space = chain['xfm_path'][0]

        selected = candidates[atlas][space]
        layout = layouts[selected['dataset']]
        atlas_image = selected['image']
        atlas_labels = layout.get_nearest(atlas_image, extension='.tsv', strict=False)
        atlas_metadata_file = layout.get_nearest(atlas_image, extension='.json', strict=True)

        if not atlas_labels:
            raise FileNotFoundError(f'No TSV file found for {atlas_image}')

        atlas_metadata = None
        if atlas_metadata_file:
            with open(atlas_metadata_file) as fo:
                atlas_metadata = json.load(fo)

        atlas_cache[atlas] = {
            'dataset': selected['dataset'],
            'image': atlas_image,
            'labels': atlas_labels,
            'metadata': atlas_metadata,
            'space': space,
            **chain,
        }

    for atlas_info in atlas_cache.values():
        if not atlas_info['labels']:
            errors.append(f'No TSV file found for {atlas_info["image"]}')
            continue

        # Check the contents of the labels file
        df = pd.read_table(atlas_info['labels'])
        if 'name' not in df.columns:
            errors.append(f"'name' column not found in {atlas_info['labels']}")

        if 'index' not in df.columns:
            errors.append(f"'index' column not found in {atlas_info['labels']}")

    if errors:
        error_str = '\n\t'.join(errors)
        raise ValueError(f'Errors found in atlas collection:\n\t{error_str}')

    return atlas_cache


def _atlas_space(atlas_file):
    """Name the space of an atlas image: its template, with any cohort as ``+<cohort>``.

    The entities are the ones PyBIDS parsed, because BIDS-Atlas allows the cohort to be
    given by a ``cohort-<label>`` folder alone.
    """
    entities = atlas_file.get_entities()
    template = entities['template']
    cohort = entities.get('cohort')
    return f'{template}+{cohort}' if cohort else template


def _chain_atlas_to_anat(atlas, spaces, xfm_edges):
    """Choose which copy of an atlas to use, and the transforms that bring it to the anatomy.

    Parameters
    ----------
    atlas : :obj:`str`
        The name of the atlas, for messages.
    spaces : :obj:`list` of :obj:`str`
        The spaces the atlas is available in.
    xfm_edges : :obj:`list` of :class:`~qsirecon.utils.transforms.Xfm`
        The template transforms in the input dataset.

    Returns
    -------
    chain : :obj:`dict`
        ``xfm_path`` is the list of spaces from the chosen copy's to the anatomical one.
        ``xfm_to_anat`` is the list of transform files in the order ANTs takes them.

    Raises
    ------
    ValueError
        If no copy of the atlas can be brought into the anatomical space.
    """
    from qsirecon.utils.transforms import (
        ANAT_SPACE,
        NoTransformChainError,
        chain_files,
        fetch_templateflow_xfm,
        find_transform_chain,
        templateflow_edges,
    )

    def _reachable(edges):
        paths = {}
        for space in spaces:
            try:
                paths[space] = find_transform_chain(
                    edges,
                    space,
                    ANAT_SPACE,
                    bidirectional=False,
                )
            except NoTransformChainError:
                continue

        return paths

    # A copy in a template the dataset was normalized to cannot be beaten,
    # so TemplateFlow is only consulted when there is none.
    edges = list(xfm_edges)
    paths = _reachable(edges)
    if not paths:
        edges = [*xfm_edges, *templateflow_edges()]
        paths = _reachable(edges)

    normalized_to = sorted({xfm.frm for xfm in xfm_edges if xfm.to == ANAT_SPACE})
    if not paths:
        raise ValueError(
            f'No chain of transforms brings the atlas {atlas} (available in '
            f'{", ".join(spaces)}) into the anatomical space. The input dataset has '
            f'transforms from: {", ".join(normalized_to) or "no template"}.'
        )

    normalized_templates = {name.partition('+')[0] for name in normalized_to}
    space = min(
        paths,
        key=lambda name: (
            len(paths[name]),
            name.partition('+')[0] not in normalized_templates,
            name,
        ),
    )
    path = paths[space]
    if len(path) > 2:
        LOGGER.warning(
            f'The atlas {atlas} is in {space}, which the input dataset has no transform from. '
            f'It will be warped into the space of the DWI through {" -> ".join(path)}, using '
            'template-to-template transforms from TemplateFlow. Each transform in the chain '
            'adds registration error, so check the alignment of the atlas in the outputs.'
        )

    xfm_to_anat = [
        fetch_templateflow_xfm(xfm) if xfm.source == 'templateflow' else xfm.path
        for xfm in chain_files(edges, path)
    ]
    return {'xfm_path': path, 'xfm_to_anat': xfm_to_anat}
