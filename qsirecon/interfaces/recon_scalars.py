#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Classes that collect scalar images and metadata from Recon Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import itertools
import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
from bids.layout import parse_file_entities
from nilearn.maskers import NiftiLabelsMasker
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    Undefined,
    isdefined,
    traits,
)

from ..utils.bids import _get_bidsuris
from ..utils.misc import deep_update_dict, load_yaml
from .bids import get_recon_output_name
from qsirecon.data import load as load_data


class ReconScalarsInputSpec(BaseInterfaceInputSpec):
    source_file = File(exists=True, mandatory=True)
    qsirecon_suffix = traits.Str(mandatory=True)
    model_info = traits.Dict()
    model_name = traits.Str()
    dismiss_entities = traits.List([], usedefault=True)


class ReconScalarsOutputSpec(TraitedSpec):
    scalar_info = traits.List()


class ReconScalars(SimpleInterface):
    input_spec = ReconScalarsInputSpec
    output_spec = ReconScalarsOutputSpec
    scalar_metadata = {}
    _ignore_traits = (
        "model_name",
        "qsirecon_suffix",
        "scalar_metadata",
        "model_info",
        "source_file",
        "dismiss_entities",
    )

    def __init__(self, from_file=None, resource_monitor=None, **inputs):
        # Get self._results defined
        super().__init__(from_file=from_file, resource_monitor=resource_monitor, **inputs)

        # Check that the input_spec matches the scalar_metadata
        self._validate_scalar_metadata()

    def _validate_scalar_metadata(self):
        for input_key in self.inputs.editable_traits():
            if (input_key in self._ignore_traits) or input_key.endswith("_metadata"):
                continue

            if input_key not in self.scalar_metadata:
                raise Exception(
                    f"No entry found for {input_key} in ``scalar_metadata`` in this class."
                )

            # Check that BIDS attributes are defined
            if "bids" not in self.scalar_metadata[input_key]:
                raise Exception(f"Missing BIDS entities for {input_key}")

    def _run_interface(self, runtime):
        results = []
        inputs = self.inputs.get()

        # Get the BIDS info from the source file
        source_file_bids = parse_file_entities(self.inputs.source_file)
        dismiss_entities = self.inputs.dismiss_entities + ["extension", "suffix"]
        source_file_bids = {k: v for k, v in source_file_bids.items() if k not in dismiss_entities}

        file_traits = [
            name for name in self.inputs.editable_traits() if name not in self._ignore_traits
        ]
        file_traits = [name for name in file_traits if not name.endswith("_metadata")]

        for input_name in file_traits:
            if not isdefined(inputs[input_name]):
                continue

            # Get the run-specific metadata for the scalar file
            metadata_name = input_name + "_metadata"
            metadata = inputs.get(metadata_name, {})
            # account for Undefined or None values
            metadata = metadata or {}

            result = self.scalar_metadata[input_name].copy()
            result["path"] = os.path.abspath(inputs[input_name])
            result["qsirecon_suffix"] = self.inputs.qsirecon_suffix
            result["variable_name"] = input_name
            result["source_file"] = self.inputs.source_file
            # Update the BIDS with the source file
            bids_overlap = set(source_file_bids.keys()).intersection(result["bids"].keys())
            if bids_overlap:
                raise Exception(
                    f"BIDS fields for {input_name} conflict with source file BIDS {bids_overlap}"
                )

            # Update the metadata with the run-specific metadata.
            # This is done recursively across all levels of the metadata dictionary.
            deep_update_dict(result["metadata"], metadata)

            results.append(result)

        self._results["scalar_info"] = results
        return runtime


class _ReconScalarsTableSplitterDataSinkInputSpec(BaseInterfaceInputSpec):
    source_file = File()
    base_directory = File()
    resampled_files = InputMultiObject(
        File(exists=True),
        desc=(
            "Resampled scalar files. "
            "This field is not used, but we keep it so that the files won't be "
            "automatically deleted by Nipype."
        ),
    )
    recon_scalars = InputMultiObject(traits.Any())
    compress = traits.Bool(True, usedefault=True)
    dismiss_entities = traits.List([], usedefault=True)
    infer_suffix = traits.Bool(False, usedefault=True)
    summary_tsv = File(exists=True, mandatory=True, desc="tsv of combined scalar summaries")
    metadata = traits.Dict(mandatory=False, desc="list of metadata dictionaries")
    suffix = traits.Str(mandatory=True)


class ReconScalarsTableSplitterDataSink(SimpleInterface):
    input_spec = _ReconScalarsTableSplitterDataSinkInputSpec
    _always_run = True

    def _run_interface(self, runtime):
        summary_df = pd.read_table(self.inputs.summary_tsv)
        for qsirecon_suffix, group_df in summary_df.groupby("qsirecon_suffix"):
            # reset the index for this df
            group_df.reset_index(drop=True, inplace=True)

            qsirecon_suffixed_tsv = get_recon_output_name(
                base_dir=self.inputs.base_directory,
                source_file=group_df.loc[0, "source_file"],
                derivative_file=self.inputs.summary_tsv,
                output_bids_entities={
                    "suffix": self.inputs.suffix,
                    "bundles": group_df.loc[0, "bundle_source"],
                },
                qsirecon_suffix=qsirecon_suffix,
                dismiss_entities=self.inputs.dismiss_entities,
            )
            output_dir = os.path.dirname(qsirecon_suffixed_tsv)
            os.makedirs(output_dir, exist_ok=True)
            group_df.to_csv(qsirecon_suffixed_tsv, index=False, sep="\t")

        return runtime


class _ParcellationTableSplitterDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc="Path to the base directory for storing data.",
    )
    compress = traits.Bool(False, usedefault=True)
    dismiss_entities = traits.List([], usedefault=True)
    in_file = File(exists=True, mandatory=True, desc="tsv of combined scalar summaries")
    meta_dict = traits.Dict(mandatory=False, desc="metadata dictionary")
    source_file = File(
        exists=False,
        mandatory=True,
        desc="the source file(s) to extract entities from",
    )
    seg = traits.Str(mandatory=True, desc="the name of the segmentation")
    suffix = traits.Str("dwimap", usedefault=True, desc="the suffix of the parcellated data")
    dataset_links = traits.Dict(mandatory=False, desc="dataset links")


class _ParcellationTableSplitterDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File(exists=True, desc="written file path"))
    out_meta = OutputMultiObject(File(exists=True, desc="written JSON sidecar path"))


class ParcellationTableSplitterDataSink(SimpleInterface):
    input_spec = _ParcellationTableSplitterDataSinkInputSpec
    output_spec = _ParcellationTableSplitterDataSinkOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        in_df = pd.read_table(self.inputs.in_file)
        self._results["out_file"] = []
        self._results["out_meta"] = []

        for qsirecon_suffix, group_df in in_df.groupby("qsirecon_suffix"):
            meta_dict = self.inputs.meta_dict.copy()
            if isdefined(self.inputs.dataset_links):
                dataset_links = self.inputs.dataset_links.copy()
                if qsirecon_suffix and qsirecon_suffix.lower() != "qsirecon":
                    out_path = os.path.join(
                        self.inputs.base_directory,
                        "derivatives",
                        f"qsirecon-{qsirecon_suffix}",
                    )
                    # We only have access to the base QSIRecon dataset's dataset_links,
                    # so we need to update the dictionary here.
                    dataset_links["qsirecon"] = self.inputs.base_directory
                    # XXX: This ignores other qsirecon_suffix datasets,
                    # but hopefully there won't be any inputs to this node from other datasets.
                else:
                    out_path = self.inputs.base_directory

                # Traverse dictionary, looking for any keys named "Sources".
                # When found, replace the value with a BIDS URI.
                for key, value in meta_dict.items():
                    if key == "Sources":
                        meta_dict[key] = _get_bidsuris(
                            value,
                            dataset_links,
                            out_path,
                        )

            # reset the index for this df
            group_df.reset_index(drop=True, inplace=True)

            qsirecon_suffixed_tsv = get_recon_output_name(
                base_dir=self.inputs.base_directory,
                source_file=self.inputs.source_file,
                derivative_file=self.inputs.in_file,
                output_bids_entities={
                    "seg": self.inputs.seg,
                    "suffix": self.inputs.suffix,
                },
                qsirecon_suffix=qsirecon_suffix,
                dismiss_entities=self.inputs.dismiss_entities,
            )
            output_dir = os.path.dirname(qsirecon_suffixed_tsv)
            os.makedirs(output_dir, exist_ok=True)
            group_df.to_csv(qsirecon_suffixed_tsv, index=False, sep="\t", na_rep="n/a")
            out_meta = qsirecon_suffixed_tsv.replace(".tsv", ".json")
            with open(out_meta, "w") as fobj:
                json.dump(meta_dict, fobj, indent=4, sort_keys=True)

            self._results["out_file"].append(qsirecon_suffixed_tsv)
            self._results["out_meta"].append(out_meta)

        return runtime


# Scalars produced in the TORTOISE recon workflow
tortoise_scalars = load_yaml(load_data("scalars/tortoise.yaml"))


class _TORTOISEReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in tortoise_scalars:
    _TORTOISEReconScalarInputSpec.add_class_trait(input_name, File(exists=True))
    if input_name.endswith("_file"):
        _TORTOISEReconScalarInputSpec.add_class_trait(
            input_name + "_metadata",
            traits.Dict(),
        )


class TORTOISEReconScalars(ReconScalars):
    input_spec = _TORTOISEReconScalarInputSpec
    scalar_metadata = tortoise_scalars


# Scalars produced in the AMICO recon workflow
amico_scalars = load_yaml(load_data("scalars/amico_noddi.yaml"))


class _AMICOReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in amico_scalars:
    _AMICOReconScalarInputSpec.add_class_trait(input_name, File(exists=True))
    if input_name.endswith("_file"):
        _AMICOReconScalarInputSpec.add_class_trait(
            input_name + "_metadata",
            traits.Dict(),
        )


class AMICOReconScalars(ReconScalars):
    input_spec = _AMICOReconScalarInputSpec
    scalar_metadata = amico_scalars


# Scalars produced by DSI Studio
dsistudio_scalars = load_yaml(load_data("scalars/dsistudio_gqi.yaml"))


class _DSIStudioReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dsistudio_scalars:
    _DSIStudioReconScalarInputSpec.add_class_trait(input_name, File(exists=True))
    if input_name.endswith("_file"):
        _DSIStudioReconScalarInputSpec.add_class_trait(
            input_name + "_metadata",
            traits.Dict(),
        )


class DSIStudioReconScalars(ReconScalars):
    input_spec = _DSIStudioReconScalarInputSpec
    scalar_metadata = dsistudio_scalars


dipy_dki_scalars = load_yaml(load_data("scalars/dipy_dki.yaml"))


class _DIPYDKIReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dipy_dki_scalars:
    _DIPYDKIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))
    if input_name.endswith("_file"):
        _DIPYDKIReconScalarInputSpec.add_class_trait(
            input_name + "_metadata",
            traits.Dict(),
        )


class DIPYDKIReconScalars(ReconScalars):
    input_spec = _DIPYDKIReconScalarInputSpec
    scalar_metadata = dipy_dki_scalars


# DIPY implementation of MAPMRI
dipy_mapmri_scalars = load_yaml(load_data("scalars/dipy_mapmri.yaml"))


class _DIPYMAPMRIReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dipy_mapmri_scalars:
    _DIPYMAPMRIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))
    _DIPYMAPMRIReconScalarInputSpec.add_class_trait(f"{input_name}_metadata", traits.Dict())


class DIPYMAPMRIReconScalars(ReconScalars):
    input_spec = _DIPYMAPMRIReconScalarInputSpec
    scalar_metadata = dipy_mapmri_scalars


# Same as DIPY implementation of 3dSHORE, but with brainsuite bases
brainsuite_3dshore_scalars = load_yaml(load_data("scalars/brainsuite_3dshore.yaml"))


class _BrainSuite3dSHOREReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in brainsuite_3dshore_scalars:
    _BrainSuite3dSHOREReconScalarInputSpec.add_class_trait(input_name, File(exists=True))
    if input_name.endswith("_file"):
        _BrainSuite3dSHOREReconScalarInputSpec.add_class_trait(
            input_name + "_metadata",
            traits.Dict(),
        )


class BrainSuite3dSHOREReconScalars(ReconScalars):
    input_spec = _BrainSuite3dSHOREReconScalarInputSpec
    scalar_metadata = brainsuite_3dshore_scalars


class _OrganizeScalarDataInputSpec(BaseInterfaceInputSpec):
    scalar_config = traits.Dict()


class _OrganizeScalarDataOutputSpec(TraitedSpec):
    scalar_file = File(exists=True)
    metadata = traits.Dict()
    model = traits.Either(
        traits.Str(),
        Undefined,
    )
    param = traits.Either(
        traits.Str(),
        Undefined,
    )
    desc = traits.Either(
        traits.Str(),
        Undefined,
        None,
    )


class OrganizeScalarData(SimpleInterface):
    input_spec = _OrganizeScalarDataInputSpec
    output_spec = _OrganizeScalarDataOutputSpec

    def _run_interface(self, runtime):
        scalar_config = self.inputs.scalar_config
        self._results["scalar_file"] = scalar_config["path"]
        self._results["metadata"] = scalar_config.get("metadata", {})
        self._results["model"] = scalar_config.get("bids", {}).get("model", Undefined)
        self._results["param"] = scalar_config.get("bids", {}).get("param", Undefined)
        self._results["desc"] = scalar_config.get("bids", {}).get("desc", None)

        return runtime


class _DisorganizeScalarDataInputSpec(BaseInterfaceInputSpec):
    scalar_config = traits.Dict()
    scalar_file = File(exists=True)


class _DisorganizeScalarDataOutputSpec(TraitedSpec):
    scalar_config = traits.Dict()


class DisorganizeScalarData(SimpleInterface):
    input_spec = _DisorganizeScalarDataInputSpec
    output_spec = _DisorganizeScalarDataOutputSpec

    def _run_interface(self, runtime):
        scalar_config = self.inputs.scalar_config
        scalar_config["path"] = self.inputs.scalar_file
        self._results["scalar_config"] = scalar_config

        return runtime


class _ParcellateScalarsInputSpec(BaseInterfaceInputSpec):
    atlas_config = traits.Dict()
    scalars_config = traits.List(traits.Dict())
    brain_mask = File(exists=True)
    mapping_metadata = traits.Dict(
        desc="Info about the upstream workflow that created the anatomical mapping units",
    )
    scalars_from = InputMultiObject(traits.Str())


class _ParcellateScalarsOutputSpec(TraitedSpec):
    parcellated_scalar_tsv = File(exists=True)
    metadata = traits.Dict()
    seg = traits.Str()


class ParcellateScalars(SimpleInterface):
    input_spec = _ParcellateScalarsInputSpec
    output_spec = _ParcellateScalarsOutputSpec

    def _run_interface(self, runtime):
        # Measures to extract: mean, stdev, median
        source_suffixes = set([cfg["qsirecon_suffix"] for cfg in self.inputs.scalars_config])
        if len(source_suffixes) > 1:
            raise ValueError(
                "All scalars must have the same qsirecon_suffix. "
                f"Found {source_suffixes} in {self.inputs.scalars_config}"
            )
        source_suffix = source_suffixes.pop()

        atlas_config = self.inputs.atlas_config
        assert len(atlas_config) == 1, "Only one atlas config is supported"
        atlas_name = list(atlas_config.keys())[0]
        atlas_config = atlas_config[atlas_name]
        atlas_file = atlas_config["dwi_resolution_file"]
        atlas_labels_file = atlas_config["labels"]
        self._results["seg"] = atlas_name

        # Fix any nonsequential values or mismatch between atlas and DataFrame.
        atlas_labels_df = pd.read_table(atlas_labels_file, index_col="index")
        atlas_img, atlas_labels_df = _sanitize_nifti_atlas(atlas_file, atlas_labels_df)

        node_labels = atlas_labels_df["label"].tolist()

        # Build empty DataFrame
        columns = ["node", "scalar", "qsirecon_suffix", "mean", "stdev", "median"]
        scalar_labels = [
            scalar_config["bids"]["param"] for scalar_config in self.inputs.scalars_config
        ]
        scalar_labels.append("coverage")
        parcellated_data = pd.DataFrame(
            columns=columns,
            data=list(
                itertools.product(
                    *[node_labels, scalar_labels, [source_suffix], [np.nan], [np.nan], [np.nan]]
                )
            ),
        )

        # prepend "background" to node labels to satisfy NiftiLabelsMasker
        # The background "label" won't be present in the output file.
        masker_input_labels = ["background"] + node_labels

        # Before anything, we need to measure coverage
        atlas_img_bin = nb.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.uint8),
            atlas_img.affine,
            atlas_img.header,
        )

        sum_masker_masked = NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=masker_input_labels,
            background_label=0,
            mask_img=self.inputs.brain_mask,
            smoothing_fwhm=None,
            standardize=False,
            strategy="sum",
            resampling_target=None,  # they should be in the same space/resolution already
        )
        sum_masker_unmasked = NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=masker_input_labels,
            background_label=0,
            smoothing_fwhm=None,
            standardize=False,
            strategy="sum",
            resampling_target=None,  # they should be in the same space/resolution already
        )
        n_voxels_in_masked_parcels = sum_masker_masked.fit_transform(atlas_img_bin)
        n_voxels_in_parcels = sum_masker_unmasked.fit_transform(atlas_img_bin)
        parcel_coverage = np.squeeze(n_voxels_in_masked_parcels / n_voxels_in_parcels)

        n_nodes = len(node_labels)
        masker_labels = sum_masker_masked.labels_[:]
        n_found_nodes = len(masker_labels)

        # Region indices in the atlas may not be sequential, so we map them to sequential ints.
        seq_mapper = {idx: i for i, idx in enumerate(atlas_labels_df["sanitized_index"].tolist())}

        if n_found_nodes != n_nodes:  # parcels lost by warping/downsampling atlas
            # Fill in any missing nodes in the array with NaNs.
            new_scalar_arr = np.full(
                n_nodes,
                fill_value=np.nan,
                dtype=parcel_coverage.dtype,
            )
            for j_col in range(parcel_coverage.size):
                label_col = seq_mapper[masker_labels[j_col]]
                new_scalar_arr[label_col] = parcel_coverage[j_col]

            parcel_coverage = new_scalar_arr
            del new_scalar_arr

        for i_node, coverage in enumerate(parcel_coverage):
            node_label = node_labels[i_node]
            parcellated_data.loc[
                (
                    (parcellated_data["node"] == node_label)
                    & (parcellated_data["scalar"] == "coverage")
                ),
                "mean",
            ] = coverage

        self._results["metadata"] = {
            "Sources": [atlas_file, self.inputs.brain_mask],
            "scalar": {
                "Description": "The scalar map from which values are extracted.",
                "Levels": {
                    "coverage": (
                        "The percent (0 - 1) of voxels in the parcel that are "
                        "within the brain mask. "
                        "Only rendered in the 'mean' column."
                    ),
                },
            },
            "node": {
                "Description": "The node label from the atlas.",
            },
            "qsirecon_suffix": {
                "Description": "The QSIRecon sub-dataset from which the scalar was extracted.",
            },
            "mean": {
                "Description": "The unweighted mean of the scalar values in the parcel.",
            },
            "stdev": {
                "Description": "The standard deviation of the scalar values in the parcel.",
            },
            "median": {
                "Description": "The median of the scalar values in the parcel.",
            },
        }
        for scalar_config in self.inputs.scalars_config:
            scalar_file = scalar_config["path"]
            scalar_img = nb.load(scalar_file)
            if scalar_img.ndim != 3:
                print(f"Scalar {scalar_file} is not 3D, skipping")
                continue

            scalar_desc = scalar_config.get("metadata", {}).get("Description", "")
            scalar_param = scalar_config.get("bids", {}).get("param", None)
            self._results["metadata"]["scalar"]["Levels"][scalar_param] = scalar_desc

            # Parcellate the scalar file with the atlas
            measures = {"mean": "mean", "stdev": "standard_deviation", "median": "median"}
            for col, metric in measures.items():
                masker = NiftiLabelsMasker(
                    labels_img=atlas_img,
                    labels=masker_input_labels,
                    background_label=0,
                    mask_img=self.inputs.brain_mask,
                    smoothing_fwhm=None,
                    standardize=False,
                    resampling_target=None,
                    strategy=metric,
                )
                scalar_arr = np.squeeze(masker.fit_transform(scalar_img))
                if n_found_nodes != n_nodes:  # parcels lost by warping/downsampling atlas
                    # Fill in any missing nodes in the array with NaNs.
                    new_scalar_arr = np.full(
                        n_nodes,
                        fill_value=np.nan,
                        dtype=scalar_arr.dtype,
                    )
                    for j_col in range(scalar_arr.size):
                        label_col = seq_mapper[masker_labels[j_col]]
                        new_scalar_arr[label_col] = scalar_arr[j_col]

                    scalar_arr = new_scalar_arr
                    del new_scalar_arr

                for i_node, scalar in enumerate(scalar_arr):
                    node_label = node_labels[i_node]
                    parcellated_data.loc[
                        (parcellated_data["node"] == node_label)
                        & (parcellated_data["scalar"] == scalar_param),
                        col,
                    ] = scalar

            self._results["metadata"]["Sources"].append(scalar_file)

        # Save the parcellated data to a tsv
        self._results["parcellated_scalar_tsv"] = os.path.abspath("parcellated_scalar_tsv.tsv")
        parcellated_data.to_csv(
            self._results["parcellated_scalar_tsv"],
            index=False,
            sep="\t",
            na_rep="n/a",
        )

        return runtime


def _sanitize_nifti_atlas(atlas, df):
    atlas_img = nb.load(atlas)
    atlas_data = atlas_img.get_fdata()
    atlas_data = atlas_data.astype(np.int16)

    # Check that all labels in the DataFrame are present in the NIfTI file, and vice versa.
    if 0 in df.index:
        df = df.drop(index=[0])

    df.sort_index(inplace=True)  # ensure index is in order
    expected_values = df.index.values

    found_values = np.unique(atlas_data)
    found_values = found_values[found_values != 0]  # drop the background value
    if not np.all(np.isin(found_values, expected_values)):
        missing_values = np.setdiff1d(found_values, expected_values)
        raise ValueError(
            f"Atlas file ({atlas}) contains values that are not present in the "
            f"DataFrame: {missing_values}\n\n{df}"
        )

    # Map the labels in the DataFrame to sequential values.
    label_mapper = {value: i + 1 for i, value in enumerate(expected_values)}
    df["sanitized_index"] = [label_mapper[i] for i in df.index.values]

    # Map the values in the atlas image to sequential values.
    new_atlas_data = np.zeros(atlas_data.shape, dtype=np.int16)
    for old_value, new_value in label_mapper.items():
        new_atlas_data[atlas_data == old_value] = new_value

    new_atlas_img = nb.Nifti1Image(new_atlas_data, atlas_img.affine, atlas_img.header)

    return new_atlas_img, df
