#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loading atlases
^^^^^^^^^^^^^^^

"""

import logging

LOGGER = logging.getLogger("nipype.interface")


def collect_atlases(datasets, atlases, bids_filters={}):
    """Collect atlases from a list of BIDS-Atlas datasets.

    Selection of labels files and metadata does not leverage the inheritance principle.
    That probably won't be possible until PyBIDS supports the BIDS-Atlas extension natively.

    Parameters
    ----------
    datasets : dict of str:str or str:BIDSLayout pairs
        Dictionary of BIDS datasets to search for atlases.
    atlases : list of str
        List of atlases to collect from across the datasets.
    bids_filters : dict
        Additional filters to apply to the BIDS query.
        Only the "atlas" key is used.

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
        - "xfm_to_anat": str
            Path to the transform to get the atlas into T1w/T2w space.
    """
    import json

    import pandas as pd
    from bids.layout import BIDSLayout

    from qsirecon.data import load as load_data

    atlas_cfg = load_data("atlas_bids_config.json")
    bids_filters = bids_filters or {}

    atlas_filter = bids_filters.get("atlas", {})
    atlas_filter["suffix"] = atlas_filter.get("suffix") or "dseg"  # XCP-D only supports dsegs
    atlas_filter["extension"] = [".nii.gz", ".nii"]
    # Hardcoded spaces for now
    atlas_filter["space"] = atlas_filter.get("space") or "MNI152NLin2009cAsym"

    atlas_cache = {}
    for dataset_name, dataset_path in datasets.items():
        if not isinstance(dataset_path, BIDSLayout):
            layout = BIDSLayout(dataset_path, config=[atlas_cfg], validate=False)
        else:
            layout = dataset_path

        if layout.get_dataset_description().get("DatasetType") != "atlas":
            continue

        for atlas in atlases:
            atlas_images = layout.get(
                atlas=atlas,
                **atlas_filter,
                return_type="file",
            )
            if not atlas_images:
                continue
            elif len(atlas_images) > 1:
                bulleted_list = "\n".join([f"  - {img}" for img in atlas_images])
                LOGGER.warning(
                    f"Multiple atlas images found for {atlas} with query {atlas_filter}:\n"
                    f"{bulleted_list}\nUsing {atlas_images[0]}."
                )

            if atlas in atlas_cache:
                raise ValueError(f"Multiple datasets contain the same atlas '{atlas}'")

            atlas_image = atlas_images[0]
            atlas_labels = layout.get_nearest(atlas_image, extension=".tsv", strict=False)
            atlas_metadata_file = layout.get_nearest(atlas_image, extension=".json", strict=True)

            if not atlas_labels:
                raise FileNotFoundError(f"No TSV file found for {atlas_image}")

            atlas_metadata = None
            if atlas_metadata_file:
                with open(atlas_metadata_file, "r") as fo:
                    atlas_metadata = json.load(fo)

            atlas_cache[atlas] = {
                "dataset": dataset_name,
                "image": atlas_image,
                "labels": atlas_labels,
                "metadata": atlas_metadata,
            }

    errors = []
    for atlas in atlases:
        if atlas not in atlas_cache:
            LOGGER.warning(f"No atlas images found for {atlas} with query {atlas_filter}")
            errors.append(f"No atlas images found for {atlas} with query {atlas_filter}")

    for atlas, atlas_info in atlas_cache.items():
        if not atlas_info["labels"]:
            errors.append(f"No TSV file found for {atlas_info['image']}")
            continue

        # Check the contents of the labels file
        df = pd.read_table(atlas_info["labels"])
        if "label" not in df.columns:
            errors.append(f"'label' column not found in {atlas_info['labels']}")

        if "index" not in df.columns:
            errors.append(f"'index' column not found in {atlas_info['labels']}")

    if errors:
        error_str = "\n\t".join(errors)
        raise ValueError(f"Errors found in atlas collection:\n\t{error_str}")

    return atlas_cache
