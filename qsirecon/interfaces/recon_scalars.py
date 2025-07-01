#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Classes that collect scalar images and metadata from Recon Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op

import pandas as pd
import nibabel as nb
import numpy as np
from bids.layout import parse_file_entities
from nilearn.maskers import NiftiLabelsMasker
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    Undefined,
    isdefined,
    traits,
)

from ..utils.misc import load_yaml
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
            if input_key in self._ignore_traits:
                continue
            if input_key not in self.scalar_metadata:
                raise Exception(
                    f"No entry found for {input_key} in ``scalar_metadata`` in this class."
                )

            # Check that BIDS attributes are defined
            if "bids" not in self.scalar_metadata[input_key]:
                raise Exception(f"Missing BIDS metadata for {input_key}")

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

        for input_name in file_traits:
            if not isdefined(inputs[input_name]):
                continue
            result = self.scalar_metadata[input_name].copy()
            result["path"] = op.abspath(inputs[input_name])
            result["qsirecon_suffix"] = self.inputs.qsirecon_suffix
            result["variable_name"] = input_name
            result["source_file"] = self.inputs.source_file
            # Update the BIDS with the source file
            bids_overlap = set(source_file_bids.keys()).intersection(result["bids"].keys())
            if bids_overlap:
                raise Exception(
                    f"BIDS fields for {input_name} conflict with source file BIDS {bids_overlap}"
                )
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
            output_dir = op.dirname(qsirecon_suffixed_tsv)
            os.makedirs(output_dir, exist_ok=True)
            group_df.to_csv(qsirecon_suffixed_tsv, index=False, sep="\t")

        return runtime


# Scalars produced in the TORTOISE recon workflow
tortoise_scalars = load_yaml(load_data("scalars/tortoise.yaml"))


class _TORTOISEReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in tortoise_scalars:
    _TORTOISEReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class TORTOISEReconScalars(ReconScalars):
    input_spec = _TORTOISEReconScalarInputSpec
    scalar_metadata = tortoise_scalars


# Scalars produced in the AMICO recon workflow
amico_scalars = load_yaml(load_data("scalars/amico_noddi.yaml"))


class _AMICOReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in amico_scalars:
    _AMICOReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class AMICOReconScalars(ReconScalars):
    input_spec = _AMICOReconScalarInputSpec
    scalar_metadata = amico_scalars


# Scalars produced by DSI Studio
dsistudio_scalars = load_yaml(load_data("scalars/dsistudio_gqi.yaml"))


class _DSIStudioReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dsistudio_scalars:
    _DSIStudioReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class DSIStudioReconScalars(ReconScalars):
    input_spec = _DSIStudioReconScalarInputSpec
    scalar_metadata = dsistudio_scalars


dipy_dki_scalars = load_yaml(load_data("scalars/dipy_dki.yaml"))


class _DIPYDKIReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dipy_dki_scalars:
    _DIPYDKIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class DIPYDKIReconScalars(ReconScalars):
    input_spec = _DIPYDKIReconScalarInputSpec
    scalar_metadata = dipy_dki_scalars


# DIPY implementation of MAPMRI
dipy_mapmri_scalars = load_yaml(load_data("scalars/dipy_mapmri.yaml"))


class _DIPYMAPMRIReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dipy_mapmri_scalars:
    _DIPYMAPMRIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class DIPYMAPMRIReconScalars(ReconScalars):
    input_spec = _DIPYMAPMRIReconScalarInputSpec
    scalar_metadata = dipy_mapmri_scalars


# Same as DIPY implementation of 3dSHORE, but with brainsuite bases
brainsuite_3dshore_scalars = load_yaml(load_data("scalars/brainsuite_3dshore.yaml"))


class _BrainSuite3dSHOREReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in brainsuite_3dshore_scalars:
    _BrainSuite3dSHOREReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


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


class _ParcellateScalarsInputSpec(BaseInterfaceInputSpec):
    atlas_config = traits.Dict()
    scalars_config = traits.List(traits.Dict())
    brain_mask = File(exists=True)
    mapping_metadata = traits.Dict(
        desc="Info about the upstream workflow that created the anatomical mapping units",
    )


class _ParcellateScalarsOutputSpec(TraitedSpec):
    parcellated_scalar_tsv = File(exists=True)
    metadata_list = traits.List(traits.Dict())
    seg = traits.Str()


class ParcellateScalars(SimpleInterface):
    input_spec = _ParcellateScalarsInputSpec
    output_spec = _ParcellateScalarsOutputSpec

    def _run_interface(self, runtime):
        source_suffix = self.inputs.mapping_metadata.get("qsirecon_suffix", "QSIRecon")

        atlas_file = self.inputs.atlas_config["path"]
        atlas_labels_file = self.inputs.atlas_config["labels"]
        self._results["seg"] = self.inputs.atlas_config["name"]

        # Fix any nonsequential values or mismatch between atlas and DataFrame.
        atlas_labels_df = pd.read_table(atlas_labels_file)
        atlas_img, atlas_labels_df = _sanitize_nifti_atlas(atlas_file, atlas_labels_df)

        atlas_labels_df["index"] = atlas_labels_df["index"].astype(int)
        if 0 in atlas_labels_df["index"].values:
            atlas_labels_df = atlas_labels_df.loc[atlas_labels_df["index"] != 0]

        node_labels = atlas_labels_df["label"].tolist()
        # prepend "background" to node labels to satisfy NiftiLabelsMasker
        # The background "label" won't be present in the output file.
        masker_labels = ["background"] + node_labels

        # Before anything, we need to measure coverage
        atlas_img_bin = nb.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.uint8),
            atlas_img.affine,
            atlas_img.header,
        )

        sum_masker_masked = NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=masker_labels,
            background_label=0,
            mask_img=self.inputs.brain_mask,
            smoothing_fwhm=None,
            standardize=False,
            strategy='sum',
            resampling_target=None,  # they should be in the same space/resolution already
        )
        sum_masker_unmasked = NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=masker_labels,
            background_label=0,
            smoothing_fwhm=None,
            standardize=False,
            strategy='sum',
            resampling_target=None,  # they should be in the same space/resolution already
        )
        n_voxels_in_masked_parcels = sum_masker_masked.fit_transform(atlas_img_bin)
        n_voxels_in_parcels = sum_masker_unmasked.fit_transform(atlas_img_bin)
        parcel_coverage = np.squeeze(n_voxels_in_masked_parcels / n_voxels_in_parcels)
        parcel_coverage_series = pd.Series(
            data=parcel_coverage,
            index=node_labels,
        )
        parcel_coverage_series['qsirecon_suffix'] = 'QSIRecon'
        n_nodes = len(node_labels)
        n_found_nodes = parcel_coverage.size

        parcellated_data = {}
        for scalar_config in self.inputs.scalars_config:
            scalar_file = scalar_config["path"]
            scalar_img = nb.load(scalar_file)
            if scalar_img.ndim != 3:
                print(f"Scalar {scalar_file} is not 3D, skipping")
                continue

            scalar_model = scalar_config["model"]
            scalar_param = scalar_config["param"]
            scalar_desc = scalar_config["desc"]

            scalar_name = f"model-{scalar_model}_param-{scalar_param}_desc-{scalar_desc}"

            # Parcellate the scalar file with the atlas
            masker = NiftiLabelsMasker(
                labels_img=atlas_img,
                labels=masker_labels,
                background_label=0,
                mask_img=self.inputs.brain_mask,
                smoothing_fwhm=None,
                standardize=False,
                resampling_target=None,  # they should be in the same space/resolution already
            )
            scalar_arr = np.squeeze(masker.fit_transform(scalar_img))

            # Region indices in the atlas may not be sequential, so we map them to sequential ints.
            seq_mapper = {idx: i for i, idx in enumerate(atlas_labels_df['sanitized_index'].tolist())}

            if n_found_nodes != n_nodes:  # parcels lost by warping/downsampling atlas
                # Fill in any missing nodes in the timeseries array with NaNs.
                new_scalar_arr = np.full(
                    n_nodes,
                    fill_value=np.nan,
                    dtype=scalar_arr.dtype,
                )
                for col in range(scalar_arr.size):
                    label_col = seq_mapper[masker_labels[col]]
                    new_scalar_arr[label_col] = scalar_arr[col]

                scalar_arr = new_scalar_arr
                del new_scalar_arr

            scalar_series = pd.Series(
                data=scalar_arr,
                index=node_labels,
            )
            scalar_series['qsirecon_suffix'] = source_suffix
            parcellated_data[scalar_name] = scalar_series

            # Prepare metadata dictionary
            metadata = {
                "Sources": [scalar_file, atlas_file],
                "model": scalar_model,
                "param": scalar_param,
                "desc": scalar_desc,
            }

            self._results["metadata_list"].append(metadata)

        parcellated_data["coverage"] = parcel_coverage_series

        # Save the parcellated data to a tsv
        parcellated_data_df = pd.DataFrame(parcellated_data)
        parcellated_data_df.to_csv(
            self._results["parcellated_scalar_tsv"],
            index=True,
            index_label="scalar",
            sep="\t",
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
        raise ValueError('Atlas file contains values that are not present in the DataFrame.')

    # Map the labels in the DataFrame to sequential values.
    label_mapper = {value: i + 1 for i, value in enumerate(expected_values)}
    df['sanitized_index'] = [label_mapper[i] for i in df.index.values]

    # Map the values in the atlas image to sequential values.
    new_atlas_data = np.zeros(atlas_data.shape, dtype=np.int16)
    for old_value, new_value in label_mapper.items():
        new_atlas_data[atlas_data == old_value] = new_value

    new_atlas_img = nb.Nifti1Image(new_atlas_data, atlas_img.affine, atlas_img.header)

    return new_atlas_img, df
