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
from copy import deepcopy

import pandas as pd
from bids.layout import parse_file_entities
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


class OrganizeScalarData(SimpleInterface):
    input_spec = _OrganizeScalarDataInputSpec
    output_spec = _OrganizeScalarDataOutputSpec

    def _run_interface(self, runtime):
        scalar_config = self.inputs.scalar_config
        self._results["scalar_file"] = scalar_config["path"]
        self._results["metadata"] = scalar_config.get("metadata", {})
        self._results["model"] = scalar_config.get("bids", {}).get("model", Undefined)
        self._results["param"] = scalar_config.get("bids", {}).get("param", Undefined)

        return runtime
