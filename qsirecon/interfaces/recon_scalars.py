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
    isdefined,
    traits,
)

from .bids import _copy_any, get_recon_output_name


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


class _ReconScalarsDataSinkInputSpec(BaseInterfaceInputSpec):
    source_file = File()
    base_directory = File()
    resampled_files = InputMultiObject(File(exists=True))
    recon_scalars = InputMultiObject(traits.Any())
    compress = traits.Bool(True, usedefault=True)
    dismiss_entities = traits.List([], usedefault=True)
    infer_suffix = traits.Bool(False, usedefault=True)


class ReconScalarsDataSink(SimpleInterface):
    input_spec = _ReconScalarsDataSinkInputSpec
    _always_run = True

    def _run_interface(self, runtime):

        force_compress = False
        force_decompress = False
        if isdefined(self.inputs.compress):
            if self.inputs.compress:
                force_compress = True
            else:
                force_decompress = True

        for recon_scalar in self.inputs.recon_scalars:
            qsirecon_suffix = None
            if self.inputs.infer_suffix:
                qsirecon_suffix = recon_scalar["qsirecon_suffix"]

            output_filename = get_recon_output_name(
                base_dir=self.inputs.base_directory,
                source_file=self.inputs.source_file,
                derivative_file=recon_scalar["path"],
                output_bids_entities=recon_scalar["bids"],
                use_ext=True,
                dismiss_entities=self.inputs.dismiss_entities,
                qsirecon_suffix=qsirecon_suffix,
            )

            if force_decompress and output_filename.endswith(".gz"):
                output_filename = output_filename.rstrip(".gz")

            if force_compress and not output_filename.endswith(".gz"):
                output_filename += ".gz"

            output_dir = op.dirname(output_filename)
            os.makedirs(output_dir, exist_ok=True)
            _copy_any(recon_scalar["path"], output_filename)

        return runtime


class _ReconScalarsTableSplitterDataSinkInputSpec(_ReconScalarsDataSinkInputSpec):
    summary_tsv = File(exists=True, mandatory=True, desc="tsv of combined scalar summaries")
    suffix = traits.Str(mandatory=True)


class ReconScalarsTableSplitterDataSink(ReconScalarsDataSink):
    input_spec = _ReconScalarsTableSplitterDataSinkInputSpec

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
tortoise_scalars = {
    "fa_file": {
        "desc": "Fractional Anisotropy from a tensor fit",
        "bids": {"param": "fa", "model": "tensor"},
    },
    "rd_file": {
        "desc": "Radial Diffusivity from a tensor fit",
        "bids": {"param": "rd", "model": "tensor"},
    },
    "ad_file": {
        "desc": "Apparent Diffusivity from a tensor fit",
        "bids": {"param": "ad", "model": "tensor"},
    },
    "li_file": {"desc": "LI from a tensor fit", "bids": {"param": "li", "model": "tensor"}},
    "am_file": {"desc": "A0 from a tensor fit", "bids": {"param": "AM", "model": "tensor"}},
    "pa_file": {"desc": "PA from MAPMRI", "bids": {"param": "PA", "model": "mapmri"}},
    "path_file": {"desc": "PAth from MAPMRI", "bids": {"param": "PAth", "model": "mapmri"}},
    "rtop_file": {
        "desc": "Return to origin probability from MAPMRI",
        "bids": {"param": "RTOP", "model": "mapmri"},
    },
    "rtap_file": {
        "desc": "Return to axis probability from MAPMRI",
        "bids": {"param": "RTAP", "model": "mapmri"},
    },
    "rtpp_file": {
        "desc": "Return to plane probability from MAPMRI",
        "bids": {"param": "RTPP", "model": "mapmri"},
    },
    "ng_file": {"desc": "Non-Gaussianity from MAPMRI", "bids": {"param": "NG", "model": "mapmri"}},
    "ngpar_file": {
        "desc": "Non-Gaussianity parallel from MAPMRI",
        "bids": {"param": "NGpar", "model": "mapmri"},
    },
    "ngperp_file": {
        "desc": "Non-Gaussianity perpendicular from MAPMRI",
        "bids": {"param": "NGperp", "model": "mapmri"},
    },
}


class _TORTOISEReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in tortoise_scalars:
    _TORTOISEReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class TORTOISEReconScalars(ReconScalars):
    input_spec = _TORTOISEReconScalarInputSpec
    scalar_metadata = tortoise_scalars


# Scalars produced in the AMICO recon workflow
amico_scalars = {
    "icvf_image": {
        "desc": "Intracellular volume fraction from NODDI",
        "bids": {"param": "icvf", "model": "noddi"},
    },
    "isovf_image": {
        "desc": "Isotropic volume fraction from NODDI",
        "bids": {"param": "isovf", "model": "noddi"},
    },
    "od_image": {"desc": "OD from NODDI", "bids": {"param": "od", "model": "noddi"}},
    "directions_image": {
        "desc": "Peak directions from NODDI",
        "reorient_on_resample": True,
        "bids": {"param": "direction", "model": "noddi"},
    },
}


class _AMICOReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in amico_scalars:
    _AMICOReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class AMICOReconScalars(ReconScalars):
    input_spec = _AMICOReconScalarInputSpec
    scalar_metadata = amico_scalars


# Scalars produced by DSI Studio
dsistudio_scalars = {
    "qa_file": {
        "desc": "Fractional Anisotropy from a tensor fit",
        "bids": {"param": "qa", "model": "GQI"},
    },
    "dti_fa_file": {
        "desc": "Radial Diffusivity from a tensor fit",
        "bids": {"param": "fa", "model": "tensor"},
    },
    "txx_file": {"desc": "Tensor fit txx", "bids": {"param": "txx", "model": "tensor"}},
    "txy_file": {"desc": "Tensor fit txy", "bids": {"param": "txy", "model": "tensor"}},
    "txz_file": {"desc": "Tensor fit txz", "bids": {"param": "txz", "model": "tensor"}},
    "tyy_file": {"desc": "Tensor fit tyy", "bids": {"param": "tyy", "model": "tensor"}},
    "tyz_file": {"desc": "Tensor fit tyz", "bids": {"param": "tyz", "model": "tensor"}},
    "tzz_file": {"desc": "Tensor fit tzz", "bids": {"param": "tzz", "model": "tensor"}},
    "rd1_file": {"desc": "RD1", "bids": {"param": "rd1", "model": "RDI"}},
    "rd2_file": {"desc": "RD2", "bids": {"param": "rd2", "model": "RDI"}},
    "ha_file": {"desc": "HA", "bids": {"param": "ha", "model": "tensor"}},
    "md_file": {"desc": "Mean Diffusivity", "bids": {"param": "md", "model": "tensor"}},
    "ad_file": {"desc": "AD", "bids": {"param": "ad", "model": "tensor"}},
    "rd_file": {"desc": "Radial Diffusivity", "bids": {"param": "rd", "model": "tensor"}},
    "gfa_file": {
        "desc": "Generalized Fractional Anisotropy",
        "bids": {"model": "GQI", "param": "gfa"},
    },
    "iso_file": {"desc": "Isotropic Diffusion", "bids": {"model": "GQI", "param": "iso"}},
}


class _DSIStudioReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dsistudio_scalars:
    _DSIStudioReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class DSIStudioReconScalars(ReconScalars):
    input_spec = _DSIStudioReconScalarInputSpec
    scalar_metadata = dsistudio_scalars


dipy_dki_scalars = {
    "dki_fa": {"desc": "DKI FA", "bids": {"param": "FA", "model": "tensor"}},
    "dki_md": {"desc": "DKI MD", "bids": {"param": "MD", "model": "dki"}},
    "dki_rd": {"desc": "DKI RD", "bids": {"param": "RD", "model": "dki"}},
    "dki_ad": {"desc": "DKI AD", "bids": {"param": "AD", "model": "dki"}},
    "dki_kfa": {"desc": "DKI KFA", "bids": {"param": "KFA", "model": "dki"}},
    "dki_mk": {"desc": "DKI MK", "bids": {"param": "MK", "model": "dki"}},
    "dki_ak": {"desc": "DKI AK", "bids": {"param": "AK", "model": "dki"}},
    "dki_rk": {"desc": "DKI RK", "bids": {"param": "RK", "model": "dki"}},
    "dki_mkt": {"desc": "DKI MKT", "bids": {"param": "MKT", "model": "dki"}},
}


class _DIPYDKIReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dipy_dki_scalars:
    _DIPYDKIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class DIPYDKIReconScalars(ReconScalars):
    input_spec = _DIPYDKIReconScalarInputSpec
    scalar_metadata = dipy_dki_scalars


# DIPY implementation of MAPMRI
dipy_mapmri_scalars = {
    "qiv_file": {
        "desc": "q-space inverse variance from MAPMRI",
        "bids": {"param": "QIV", "model": "mapmri"},
    },
    "msd_file": {
        "desc": "mean square displacement from MAPMRI",
        "bids": {"param": "MSD", "model": "mapmri"},
    },
    "lapnorm_file": {
        "desc": "Laplacian norm from regularized MAPMRI (MAPL)",
        "bids": {"param": "lapnorm", "model": "mapmri"},
    },
    "rtop_file": {
        "desc": "Return to origin probability from MAPMRI",
        "bids": {"param": "RTOP", "model": "mapmri"},
    },
    "rtap_file": {
        "desc": "Return to axis probability from MAPMRI",
        "bids": {"param": "RTAP", "model": "mapmri"},
    },
    "rtpp_file": {
        "desc": "Return to plane probability from MAPMRI",
        "bids": {"param": "RTPP", "model": "mapmri"},
    },
    "ng_file": {"desc": "Non-Gaussianity from MAPMRI", "bids": {"param": "NG", "model": "mapmri"}},
    "ngpar_file": {
        "desc": "Non-Gaussianity parallel from MAPMRI",
        "bids": {"param": "NGpar", "model": "mapmri"},
    },
    "ngperp_file": {
        "desc": "Non-Gaussianity perpendicular from MAPMRI",
        "bids": {"param": "NGperp", "model": "mapmri"},
    },
    "mapcoeffs_file": {
        "desc": "MAPMRI coefficients",
        "bids": {"param": "mapcoeffs", "model": "mapmri"},
    },
}


class _DIPYMAPMRIReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in dipy_mapmri_scalars:
    _DIPYMAPMRIReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class DIPYMAPMRIReconScalars(ReconScalars):
    input_spec = _DIPYMAPMRIReconScalarInputSpec
    scalar_metadata = dipy_mapmri_scalars


# Same as DIPY implementation of 3dSHORE, but with brainsuite bases
brainsuite_3dshore_scalars = deepcopy(dipy_mapmri_scalars)
brainsuite_3dshore_scalars.update(
    {
        "cnr_image": {
            "desc": "Contrast to noise ratio for 3dSHORE fit",
            "bids": {"param": "CNR", "model": "3dSHORE"},
        },
        "alpha_image": {
            "desc": "alpha used when fitting in each voxel",
            "bids": {"param": "alpha", "model": "3dSHORE"},
        },
        "r2_image": {
            "desc": "r^2 of the 3dSHORE fit",
            "bids": {"param": "rsquared", "model": "3dSHORE"},
        },
        "regularization_image": {
            "desc": "regularization of the 3dSHORE fit",
            "bids": {"param": "regularization", "model": "3dSHORE"},
        },
    }
)
for key in brainsuite_3dshore_scalars:
    brainsuite_3dshore_scalars[key]["bids"]["model"] = "3dSHORE"


class _BrainSuite3dSHOREReconScalarInputSpec(ReconScalarsInputSpec):
    pass


for input_name in brainsuite_3dshore_scalars:
    _BrainSuite3dSHOREReconScalarInputSpec.add_class_trait(input_name, File(exists=True))


class BrainSuite3dSHOREReconScalars(ReconScalars):
    input_spec = _BrainSuite3dSHOREReconScalarInputSpec
    scalar_metadata = brainsuite_3dshore_scalars
