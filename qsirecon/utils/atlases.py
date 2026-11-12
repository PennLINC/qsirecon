# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loading atlases
^^^^^^^^^^^^^^^

"""

import logging
import re
from dataclasses import dataclass

LOGGER = logging.getLogger('nipype.interface')

#: The template atlases are taken from when nothing says which one to use
DEFAULT_ATLAS_TEMPLATE = 'MNI152NLin2009cAsym'
#: The selectors an ``--atlases`` token may carry, in the order they are written out
ATLAS_SELECTOR_KEYS = ('tpl', 'cohort', 'res', 'desc', 'seg')
#: The PyBIDS names a ``--bids-filter-file`` may use for each selector
_FILTER_NAMES = {
    'tpl': ('template',),
    'cohort': ('cohort',),
    'res': ('res', 'resolution'),
    'desc': ('desc', 'description'),
    'seg': ('seg', 'segmentation'),
}
_LABEL_RE = re.compile(r'^[a-zA-Z0-9]+$')


class AtlasSelectorError(ValueError):
    """An ``--atlases`` token could not be parsed or validated."""


@dataclass(frozen=True)
class AtlasSpec:
    """One requested atlas, with the selectors that pin which copy of it is used."""

    name: str
    tpl: str | None = None
    cohort: str | None = None
    res: str | None = None
    desc: str | None = None
    seg: str | None = None

    @property
    def selectors(self) -> dict:
        """The selectors that were given, as ``{key: value}``."""
        return {
            key: getattr(self, key)
            for key in ATLAS_SELECTOR_KEYS
            if getattr(self, key) is not None
        }

    @property
    def selector_str(self) -> str:
        """The selectors as they are written on the command line, without the name."""
        return ':'.join(f'{key}-{value}' for key, value in self.selectors.items())

    def matches(self, entities: dict) -> bool:
        """Whether a file with these entities is a copy the selectors allow."""
        return all(entities.get(key) == value for key, value in self.selectors.items())

    def __str__(self) -> str:
        return ':'.join(part for part in (self.name, self.selector_str) if part)


def parse_atlas_token(token: str) -> AtlasSpec:
    """Parse one ``--atlases`` token into an :class:`AtlasSpec`.

    The grammar is the one ``--output-spaces`` uses, with the atlas name in the place of
    the template and the template as a ``tpl-`` selector::

        token   ::= atlas (":" key "-" value)*
        key     ::= "tpl" | "cohort" | "res" | "desc" | "seg"

    Every selector is matched literally against the entity of the same name in the atlas
    file, so ``res-01`` and ``res-1`` are different.
    ``res-`` only chooses a file: atlases are always resampled onto the grid of the DWI.
    """
    from qsirecon.utils import spaces

    name, *rest = token.split(':')
    if not name:
        raise AtlasSelectorError(f'"{token}" does not start with the name of an atlas.')

    selectors = {}
    pairs = spaces.split_key_values(token, rest, ATLAS_SELECTOR_KEYS, error=AtlasSelectorError)
    for key, value in pairs:
        if key in selectors:
            raise AtlasSelectorError(f'"{token}" specifies more than one {key}-.')

        if not _LABEL_RE.match(value):
            raise AtlasSelectorError(
                f'"{key}-{value}" in "{token}" is not valid: '
                'a label may only have letters and digits.'
            )

        selectors[key] = value

    res = selectors.get('res')
    if res is not None and spaces._MM_RE.match(res):
        raise AtlasSelectorError(
            f'res-{res} in "{token}" is a voxel size. In --atlases, res- only selects the '
            'atlas file with that res- label in its name: atlases are always resampled onto '
            'the grid of the preprocessed DWI. Use the label the atlas file has, for '
            'example res-01.'
        )

    tpl, cohort = selectors.get('tpl'), selectors.get('cohort')
    if cohort is not None and tpl is None:
        raise AtlasSelectorError(
            f'"{token}" names a cohort without a template. Add the template the cohort '
            f'belongs to, for example tpl-MNIInfant:cohort-{cohort}.'
        )

    if cohort == 'auto':
        raise AtlasSelectorError(
            f'cohort-auto in "{token}" is not valid in --atlases. Name the cohort, or leave '
            'off the selectors to let QSIRecon choose the copy of the atlas with the '
            'shortest chain of transforms to the anatomical space.'
        )

    # A template TemplateFlow does not know may still be one the input dataset has
    # transforms for, so only the cohorts of known templates are checked here.
    if tpl is not None and tpl in spaces._templates():
        available = spaces._cohorts(tpl)
        options = ', '.join(f'cohort-{c}' for c in available)
        if cohort is None and available:
            raise AtlasSelectorError(
                f'tpl-{tpl} in "{token}" is not fully defined: it needs a cohort. '
                f'Use one of: {options}.'
            )

        if cohort is not None and not available:
            raise AtlasSelectorError(
                f'tpl-{tpl} in "{token}" does not accept a cohort specification.'
            )

        if cohort is not None and cohort not in available:
            raise AtlasSelectorError(
                f'tpl-{tpl} in "{token}" has no cohort-{cohort}. Available cohorts are: {options}.'
            )

    return AtlasSpec(name=name, **selectors)


def parse_atlases(tokens) -> list[AtlasSpec]:
    """Parse every ``--atlases`` token and check that no atlas is requested twice.

    Outputs are named after the atlas alone (``seg-<atlas>``), so two copies of one
    atlas would be written to the same files.
    A token that is repeated exactly is one request.
    """
    specs = {}
    for token in tokens:
        spec = token if isinstance(token, AtlasSpec) else parse_atlas_token(str(token))
        previous = specs.setdefault(spec.name, spec)
        if previous != spec:
            raise AtlasSelectorError(
                f'The atlas {spec.name} was requested more than once, as "{previous}" and '
                f'"{spec}". Outputs are named after the atlas alone (seg-{spec.name}), so '
                'only one copy of an atlas can be used in a run.'
            )

    return list(specs.values())


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

    An atlas requested with selectors (``4S156Parcels:tpl-MNI152NLin2009cAsym:res-01``)
    is only taken from the copies the selectors describe.
    It is an error if there is no such copy, or if no chain of transforms brings it into
    the anatomical space.
    When the copies in one template differ in resolution and no ``res-`` selector chooses
    between them, the one with the smallest voxels is used, with a warning.

    Parameters
    ----------
    datasets : dict of str:str or str:BIDSLayout pairs
        Dictionary of BIDS datasets to search for atlases.
    atlases : list of str or :class:`AtlasSpec`
        The atlases to collect from across the datasets, as ``--atlases`` tokens:
        an atlas name, optionally followed by selectors (see :func:`parse_atlas_token`).
    bids_filters : dict
        Additional filters to apply to the BIDS query.
        Only the "atlas" key is used.
        A "template" entry restricts the templates atlases may be taken from.
        The selectors of an atlas must agree with these filters.
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
    import os

    import pandas as pd
    from bids.layout import BIDSLayout

    from qsirecon.data import load as load_data

    atlas_cfg = load_data('atlas_bids_config.json')
    bids_filters = bids_filters or {}
    specs = parse_atlases(atlases)

    atlas_filter = dict(bids_filters.get('atlas', {}))
    atlas_filter['suffix'] = atlas_filter.get('suffix') or 'dseg'  # XCP-D only supports dsegs
    atlas_filter['extension'] = ['.nii.gz', '.nii']
    if not atlas_filter.get('template'):
        # Any template will do: the one with the shortest chain of transforms is chosen
        atlas_filter.pop('template', None)

    errors = []
    # The query each atlas is looked up with
    queries = {}
    for spec in specs:
        conflict = _filter_conflict(spec, atlas_filter)
        if conflict:
            errors.append(conflict)
            continue

        query = dict(atlas_filter)
        if xfm_edges is None and 'template' not in query and spec.tpl is None:
            query['template'] = DEFAULT_ATLAS_TEMPLATE

        queries[spec.name] = query

    specs = [spec for spec in specs if spec.name in queries]
    layouts = {}
    # The copies of each atlas that may be used: {atlas: {space: {'dataset': ..., 'image': ...}}}
    candidates = {spec.name: {} for spec in specs}
    # Every copy of each atlas that was found, whatever its selectors, for error messages
    available = {spec.name: set() for spec in specs}
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
        for spec in specs:
            atlas = spec.name
            by_space = {}
            for atlas_file in layout.get(atlas=atlas, **queries[atlas]):
                entities = _atlas_entities(atlas_file)
                available[atlas].add(_describe_copy(entities))
                if spec.matches(entities):
                    by_space.setdefault(_atlas_space(atlas_file), []).append(
                        (atlas_file.path, entities.get('res')),
                    )

            for space, images in by_space.items():
                image = _pick_image(atlas, space, images)
                if space in candidates[atlas]:
                    raise ValueError(f"Multiple datasets contain the same atlas '{atlas}'")

                candidates[atlas][space] = {'dataset': dataset_name, 'image': image}

    atlas_cache = {}
    for spec in specs:
        atlas = spec.name
        if not candidates[atlas]:
            if spec.selectors and available[atlas]:
                message = (
                    f'No copy of the atlas {atlas} matches the selectors '
                    f'"{spec.selector_str}". The copies found with query {queries[atlas]} '
                    f'are: {", ".join(sorted(available[atlas]))}.'
                )
            else:
                message = f'No atlas images found for {atlas} with query {queries[atlas]}'

            LOGGER.warning(message)
            errors.append(message)
            continue

        spaces = sorted(candidates[atlas])
        chain = {}
        if xfm_edges is None:
            space = spaces[0]
            if len(spaces) > 1:
                LOGGER.warning(f'{atlas} is available in {", ".join(spaces)}. Using {space}.')
        else:
            try:
                chain = _chain_atlas_to_anat(
                    atlas,
                    spaces,
                    xfm_edges,
                    selectors=spec.selector_str,
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue

            space = chain['xfm_path'][0]

        selected = candidates[atlas][space]
        layout = layouts[selected['dataset']]
        atlas_image = selected['image']
        # The files named like the image come first: PyBIDS only compares the entities it
        # knows, which need not include every selector an atlas was requested with.
        atlas_stem = atlas_image.removesuffix('.gz').removesuffix('.nii')
        atlas_labels = layout.get_nearest(atlas_image, extension='.tsv', strict=False)
        if os.path.isfile(f'{atlas_stem}.tsv'):
            atlas_labels = f'{atlas_stem}.tsv'

        atlas_metadata_file = layout.get_nearest(atlas_image, extension='.json', strict=True)
        if os.path.isfile(f'{atlas_stem}.json'):
            atlas_metadata_file = f'{atlas_stem}.json'

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


def _atlas_entities(atlas_file):
    """List the entities of an atlas image that ``--atlases`` selectors are matched against.

    The template and cohort are the ones PyBIDS parsed (see :func:`_atlas_space`).
    The others are read from the file name, so that labels are compared as written.
    """
    parsed = atlas_file.get_entities()
    cohort = parsed.get('cohort')
    entities = {'tpl': str(parsed['template'])}
    if cohort:
        entities['cohort'] = str(cohort)

    entities.update(re.findall(r'(?:^|_)(res|desc|seg)-([a-zA-Z0-9]+)', atlas_file.filename))
    return entities


def _describe_copy(entities):
    """Spell the entities of an atlas image the way ``--atlases`` selectors are written."""
    return ':'.join(f'{key}-{entities[key]}' for key in ATLAS_SELECTOR_KEYS if entities.get(key))


def _filter_conflict(spec, atlas_filter):
    """Explain how an atlas's selectors disagree with the ``atlas`` BIDS filters, if they do."""
    for key, value in spec.selectors.items():
        for name in _FILTER_NAMES[key]:
            allowed = atlas_filter.get(name)
            if isinstance(allowed, (str, int)):
                allowed = [allowed]

            if not isinstance(allowed, (list, tuple)) or not allowed:
                continue

            if value not in [str(item) for item in allowed]:
                return (
                    f'The atlas {spec.name} was requested with {key}-{value}, but the "atlas" '
                    f'section of the --bids-filter-file sets "{name}" to '
                    f'{", ".join(str(item) for item in allowed)}. The two must agree: change '
                    'one of them, or remove the entry from the filter file.'
                )

    return None


def _pick_image(atlas, space, images):
    """Choose between the images of one atlas in one space.

    Parameters
    ----------
    atlas : :obj:`str`
        The name of the atlas, for messages.
    space : :obj:`str`
        The space the images are in, for messages.
    images : :obj:`list` of :obj:`tuple`
        The path and ``res`` label (or None) of each image.

    Returns
    -------
    image : :obj:`str`
        The image with the smallest voxels when the images differ in resolution,
        and the first one otherwise.
    """
    images = sorted(images, key=lambda image: image[0])
    if len(images) == 1:
        return images[0][0]

    bulleted_list = '\n'.join(f'  - {path}' for path, _ in images)
    if len({res for _, res in images}) == 1:
        LOGGER.warning(
            f'Multiple atlas images found for {atlas} in {space}:\n{bulleted_list}\n'
            f'Using {images[0][0]}. To choose another, add selectors to the atlas in '
            f'--atlases (for example {atlas}:desc-<label>).'
        )
        return images[0][0]

    volumes = {path: _voxel_volume(path) for path, _ in images}
    image = min(volumes, key=lambda path: (volumes[path], path))
    LOGGER.warning(
        f'The atlas {atlas} is available at more than one resolution in {space}:\n'
        f'{bulleted_list}\nUsing the one with the smallest voxels, {image}. To choose '
        f'another, add a res- selector to the atlas in --atlases (for example '
        f'{atlas}:res-<label>).'
    )
    return image


def _voxel_volume(path):
    """Read the volume of a voxel of an image, in cubic millimeters."""
    import nibabel as nb
    import numpy as np

    return float(np.prod(nb.load(path).header.get_zooms()[:3]))


def _chain_atlas_to_anat(atlas, spaces, xfm_edges, selectors=''):
    """Choose which copy of an atlas to use, and the transforms that bring it to the anatomy.

    Parameters
    ----------
    atlas : :obj:`str`
        The name of the atlas, for messages.
    spaces : :obj:`list` of :obj:`str`
        The spaces the atlas is available in.
    xfm_edges : :obj:`list` of :class:`~qsirecon.utils.transforms.Xfm`
        The template transforms in the input dataset.
    selectors : :obj:`str`
        The ``--atlases`` selectors that ``spaces`` were restricted by, for messages.

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
        requested = f', requested with the selectors "{selectors}",' if selectors else ''
        hint = ' Change the selectors, or leave them off.' if selectors else ''
        where = 'the copies that match are in' if selectors else 'available in'
        raise ValueError(
            f'No chain of transforms brings the atlas {atlas}{requested} ({where} '
            f'{", ".join(spaces)}) into the anatomical space. The input dataset has '
            f'transforms from: {", ".join(normalized_to) or "no template"}.{hint}'
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
