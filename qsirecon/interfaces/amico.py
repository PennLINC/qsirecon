#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Workflows for AMICO
~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op

import nibabel as nb
import nilearn.image as nim
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from ..utils.boilerplate import ConditionalDoc
from .converters import (
    amplitudes_to_fibgz,
    amplitudes_to_sh_mif,
    get_dsi_studio_ODF_geometry,
)

LOGGER = logging.getLogger("nipype.interface")
TAU_DEFAULT = 1.0 / (4 * np.pi**2)


class AmicoInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    big_delta = traits.Either(None, traits.Float(), usedefault=True)
    little_delta = traits.Either(None, traits.Float(), usedefault=True)
    b0_threshold = traits.CFloat(50, usedefault=True)
    # Outputs
    write_fibgz = traits.Bool(True)
    write_mif = traits.Bool(True)


class AmicoOutputSpec(TraitedSpec):
    fibgz = File()
    fod_sh_mif = File()
    extrapolated_dwi = File()
    extrapolated_bvals = File()
    extrapolated_bvecs = File()
    extrapolated_b = File()
    odf_amplitudes = File()
    odf_directions = File()


class AmicoReconInterface(SimpleInterface):
    input_spec = AmicoInputSpec
    output_spec = AmicoOutputSpec

    def _get_scheme(self, external_bvals=None, external_bvecs=None):
        little_delta = self.inputs.little_delta if isdefined(self.inputs.little_delta) else None
        big_delta = self.inputs.big_delta if isdefined(self.inputs.big_delta) else None
        bval_file = self.inputs.bval_file if external_bvals is None else external_bvals
        bvec_file = self.inputs.bvec_file if external_bvecs is None else external_bvecs
        gtab = gradient_table(
            bvals=np.loadtxt(bval_file),
            bvecs=np.loadtxt(bvec_file).T,
            b0_threshold=self.inputs.b0_threshold,
            big_delta=big_delta,
            small_delta=little_delta,
        )
        return gtab

    def _save_scalar(self, data, suffix, runtime, ref_img):
        output_fname = fname_presuffix(self.inputs.dwi_file, suffix=suffix, newpath=runtime.cwd)
        nb.Nifti1Image(data, ref_img.affine, ref_img.header).to_filename(output_fname)
        return output_fname

    def _write_external_formats(self, runtime, fit_obj, mask_img, suffix):

        if not (self.inputs.write_fibgz or self.inputs.write_mif):
            return

        # Convert to amplitudes for other software
        verts, faces = get_dsi_studio_ODF_geometry("odf8")
        num_dirs, _ = verts.shape
        hemisphere = num_dirs // 2
        x, y, z = verts[:hemisphere].T
        hs = HemiSphere(x=x, y=y, z=z)
        odf_amplitudes = nb.Nifti1Image(fit_obj.odf(hs), mask_img.affine, mask_img.header)
        output_amps_file = fname_presuffix(
            self.inputs.dwi_file, suffix=suffix + "_amp.nii.gz", newpath=runtime.cwd, use_ext=False
        )
        output_dirs_file = fname_presuffix(
            self.inputs.dwi_file, suffix=suffix + "_dirs.npy", newpath=runtime.cwd, use_ext=False
        )
        odf_amplitudes.to_filename(output_amps_file)
        np.save(output_dirs_file, verts[:hemisphere])
        self._results["odf_amplitudes"] = output_amps_file
        self._results["odf_directions"] = output_dirs_file

        if self.inputs.write_fibgz:
            output_fib_file = fname_presuffix(
                self.inputs.dwi_file, suffix=suffix + ".fib", newpath=runtime.cwd, use_ext=False
            )
            LOGGER.info("Writing DSI Studio fib file %s", output_fib_file)
            amplitudes_to_fibgz(
                odf_amplitudes, verts, faces, output_fib_file, mask_img, num_fibers=5
            )
            self._results["fibgz"] = output_fib_file

        if self.inputs.write_mif:
            output_mif_file = fname_presuffix(
                self.inputs.dwi_file, suffix=suffix + ".mif", newpath=runtime.cwd, use_ext=False
            )
            LOGGER.info("Writing sh mif file %s", output_mif_file)
            amplitudes_to_sh_mif(odf_amplitudes, verts, output_mif_file, runtime.cwd)
            self._results["fod_sh_mif"] = output_mif_file


class NODDIInputSpec(AmicoInputSpec):
    dPar = traits.Float(
        mandatory=True,
        desc=(
            "Parallel diffusivity constant in mm^/2. "
            "Passed to AMICO as dPar. "
            "Defined in the recon spec as dPar. "
            "Stored in the metadata as ParallelDiffusivity."
        ),
        doc=ConditionalDoc("The parallel diffusivity constant was set to {value} mm^2/s."),
    )
    dIso = traits.Float(
        mandatory=True,
        desc=(
            "Isotropic diffusivity constant in mm^2/s. "
            "Passed to AMICO as dIso. "
            "Defined in the recon spec as dIso. "
            "Stored in the metadata as IsotropicDiffusivity."
        ),
        doc=ConditionalDoc("The isotropic diffusivity constant was set to {value} mm^2/s."),
    )
    b0_threshold = traits.Float(
        50.0,
        usedefault=True,
        desc="Threshold below which a b-value is considered a b0.",
        doc=ConditionalDoc("All b-values were rounded to the closest {value} s/mm²."),
    )
    isExvivo = traits.Bool(
        False,
        usedefault=True,
        desc=(
            "Flag indicating whether acquired data is ex vivo or fixed tissue. "
            "Fixed tissue requires an additional dot compartment estimation."
        ),
        doc=ConditionalDoc(
            if_true=(
                "An additional dot compartment was computed "
                "to account for hindered diffusivity in fixed tissue."
            ),
        ),
    )
    doNormalize = traits.Bool(
        True,
        usedefault=True,
        desc=("Flag indicating whether to normalize the data during fitting. Default: True."),
        doc=ConditionalDoc(
            if_true=(
                "Diffusion-weighted images were normalized to the mean non-diffusion-weighted "
                "(b0) signal to reduce intensity variation across volumes. For each voxel, "
                "the mean signal intensity was computed across all b0 volumes identified in "
                "the acquisition scheme. Voxelwise normalization factors were then defined as "
                "the reciprocal of these mean b0 values. To prevent division by noise-dominated "
                "or near-zero intensities, voxels with mean b0 signal below a fixed threshold "
                "(defined as b0 x mean of all positive b0 values) were excluded from "
                "normalization by setting their factors to zero. The resulting normalization "
                "factors were applied to each diffusion-weighted volume by voxelwise "
                "multiplication, yielding data expressed as signal intensity relative "
                "to the mean b0 image."
            ),
        ),
    )
    rmse = traits.Bool(
        True,
        usedefault=True,
        desc=(
            "Flag indicating whether to compute the root mean square error "
            "(RMSE) between the measured and fitted signals. Default: True."
        ),
        doc=ConditionalDoc(
            if_true="The root mean square error (RMSE) between the measured "
            "and fitted signals was computed."
        ),
    )
    nrmse = traits.Bool(
        True,
        usedefault=True,
        desc=(
            "Flag indicating whether to compute the normalized root mean square error"
            " (NRMSE) between the measured and fitted signals. "
            "Default: True."
        ),
        doc=ConditionalDoc(
            if_true=(
                "The normalized root mean square error (NRMSE) "
                "between the measured and fitted signals was computed."
            ),
        ),
    )
    saveModulatedMaps = traits.Bool(
        True,
        usedefault=True,
        desc=("Flag indicating whether to save modulated maps for ICVF and ODI. Default: True."),
        doc=ConditionalDoc(
            if_true=(
                "Intracellular volume fraction (ICVF) and Orientation dispersion maps "
                "were multipled by the tissue fraction (1 - isotropic volume fraction) "
                "in AMICO to produce tissue fraction modulated maps (@parker2021not)."
            ),
        ),
    )
    fitMethod = traits.Enum(
        "OLS",
        "WLS",
        usedefault=True,
        desc=(
            "Fitting method to use. Options are 'OLS' (Ordinary Least Squares) "
            "or 'WLS' (Weighted Least Squares). "
            "Default: 'OLS'."
        ),
        doc=ConditionalDoc(
            "Peak directions were estimated from a diffusion tensor model "
            "using {value} fitting in DIPY (@dipy)."
        ),
    )
    num_threads = traits.Int(1, usedefault=True, nohash=True)


class NODDIOutputSpec(AmicoOutputSpec):
    directions_file = File()
    directions_file_metadata = traits.Dict()
    icvf_file = File()
    icvf_file_metadata = traits.Dict()
    od_file = File()
    od_file_metadata = traits.Dict()
    isovf_file = File()
    isovf_file_metadata = traits.Dict()
    modulated_icvf_file = File()
    modulated_icvf_file_metadata = traits.Dict()
    modulated_od_file = File()
    modulated_od_file_metadata = traits.Dict()
    rmse_file = File()
    rmse_file_metadata = traits.Dict()
    nrmse_file = File()
    nrmse_file_metadata = traits.Dict()
    config_file = File()


class NODDI(AmicoReconInterface):
    input_spec = NODDIInputSpec
    output_spec = NODDIOutputSpec

    def _run_interface(self, runtime):

        # shim the expected inputs
        shim_dir = op.join(runtime.cwd, "study/subject")
        os.makedirs(shim_dir)
        bval_file = fname_presuffix(self.inputs.bval_file, newpath=shim_dir)
        bvec_file = fname_presuffix(self.inputs.bvec_file, newpath=shim_dir)
        dwi_file = fname_presuffix(self.inputs.dwi_file, newpath=shim_dir)
        mask_file = fname_presuffix(self.inputs.mask_file, newpath=shim_dir)
        os.symlink(self.inputs.bval_file, bval_file)
        os.symlink(self.inputs.bvec_file, bvec_file)
        os.symlink(self.inputs.dwi_file, dwi_file)
        os.symlink(self.inputs.mask_file, mask_file)

        # Prevent a ton of deprecation warnings
        os.environ["KMP_WARNINGS"] = "0"
        import amico

        if hasattr(amico, "set_verbose"):
            amico.set_verbose(2)
        # Set up the AMICO evaluation
        aeval = amico.Evaluation("study", "subject")
        scheme_file = amico.util.fsl2scheme(
            bval_file, bvec_file, bStep=self.inputs.b0_threshold, flipAxes=[False, True, True]
        )
        aeval.load_data(
            dwi_filename=dwi_file,
            scheme_filename=scheme_file,
            mask_filename=mask_file,
            b0_thr=self.inputs.b0_threshold,
        )
        LOGGER.info("Fitting NODDI Model.")
        aeval.set_model("NODDI")

        # Set global configuration
        aeval.set_config("BLAS_nthreads", 1)
        aeval.set_config("nthreads", self.inputs.num_threads)
        aeval.set_config("doSaveModulatedMaps", self.inputs.saveModulatedMaps)
        aeval.set_config("doComputeRMSE", self.inputs.rmse)
        aeval.set_config("doComputeNRMSE", self.inputs.nrmse)
        aeval.set_config("doNormalize", self.inputs.doNormalize)
        aeval.set_config("DTI_fit_method", self.inputs.fitMethod)

        # Set model parameters
        aeval.model.dPar = self.inputs.dPar
        aeval.model.dIso = self.inputs.dIso
        aeval.model.isExvivo = self.inputs.isExvivo

        # Generate kernels and fit
        aeval.generate_kernels()
        aeval.load_kernels()
        aeval.fit()

        # Write the results
        aeval.save_results()
        self._results["directions_file"] = shim_dir + "/AMICO/NODDI/fit_dir.nii.gz"
        self._results["icvf_file"] = shim_dir + "/AMICO/NODDI/fit_NDI.nii.gz"
        self._results["od_file"] = shim_dir + "/AMICO/NODDI/fit_ODI.nii.gz"
        self._results["isovf_file"] = shim_dir + "/AMICO/NODDI/fit_FWF.nii.gz"
        self._results["modulated_od_file"] = shim_dir + "/AMICO/NODDI/fit_ODI_modulated.nii.gz"
        self._results["modulated_icvf_file"] = shim_dir + "/AMICO/NODDI/fit_NDI_modulated.nii.gz"
        self._results["rmse_file"] = shim_dir + "/AMICO/NODDI/fit_RMSE.nii.gz"
        self._results["nrmse_file"] = shim_dir + "/AMICO/NODDI/fit_NRMSE.nii.gz"
        self._results["config_file"] = shim_dir + "/AMICO/NODDI/config.pickle"

        return runtime

    def _list_outputs(self):
        base_metadata = {
            "Model": {
                "Parameters": {
                    "ParallelDiffusivity": self.inputs.dPar,
                    "IsotropicDiffusivity": self.inputs.dIso,
                    "IsExvivo": self.inputs.isExvivo,
                    "saveModulatedMaps": self.inputs.saveModulatedMaps,
                    "fitMethod": self.inputs.fitMethod,
                    "doNormalize": self.inputs.doNormalize,
                    "rmse": self.inputs.rmse,
                    "nrmse": self.inputs.nrmse,
                },
            },
        }
        outputs = super()._list_outputs()
        file_outputs = [name for name in self.output_spec().get() if name.endswith("_file")]
        for file_output in file_outputs:
            # Patch in model and parameter information to metadata dictionaries
            metadata_output = file_output + "_metadata"
            if metadata_output in self.output_spec().get():
                outputs[metadata_output] = base_metadata

        return outputs


class _NODDITissueFractionInputSpec(BaseInterfaceInputSpec):
    isovf_file = File(exists=True, mandatory=True)
    mask_image = File(exists=True, mandatory=True)


class _NODDITissueFractionOutputSpec(TraitedSpec):
    tf_file = File()


class NODDITissueFraction(SimpleInterface):
    input_spec = _NODDITissueFractionInputSpec
    output_spec = _NODDITissueFractionOutputSpec

    def _run_interface(self, runtime):
        isovf_file = self.inputs.isovf_file
        mask_image = self.inputs.mask_image

        tf_file = nim.math_img("(1 - isovf) * mask", isovf=isovf_file, mask=mask_image)
        out_file = fname_presuffix(isovf_file, suffix="_tf", newpath=runtime.cwd)
        tf_file.to_filename(out_file)
        self._results["tf_file"] = out_file
        return runtime
