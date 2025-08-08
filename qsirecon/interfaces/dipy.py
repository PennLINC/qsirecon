#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import shutil

import nibabel as nb
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere
from dipy.io.utils import nifti1_symmat
from dipy.reconst import dki, dki_micro, dti, mapmri
from dipy.segment.mask import median_otsu
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
from pkg_resources import resource_filename as pkgr

from ..interfaces.mrtrix import _convert_fsl_to_mrtrix
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis
from .converters import (
    amplitudes_to_fibgz,
    amplitudes_to_sh_mif,
    get_dsi_studio_ODF_geometry,
)

LOGGER = logging.getLogger("nipype.interface")
TAU_DEFAULT = 1.0 / (4 * np.pi**2)


class DipyReconInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    local_bvec_file = File(exists=True)
    big_delta = traits.Either(None, traits.Float(), usedefault=True)
    small_delta = traits.Either(None, traits.Float(), usedefault=True)
    b0_threshold = traits.CFloat(50, usedefault=True)
    # Outputs
    write_fibgz = traits.Bool(True)
    write_mif = traits.Bool(True)
    # To extrapolate
    extrapolate_scheme = traits.Enum("HCP", "ABCD", "DSIQ5")


class DipyReconOutputSpec(TraitedSpec):
    fibgz = File()
    fod_sh_mif = File()
    extrapolated_dwi = File()
    extrapolated_bvals = File()
    extrapolated_bvecs = File()
    extrapolated_b = File()
    odf_amplitudes = File()
    odf_directions = File()


class DipyReconInterface(SimpleInterface):
    input_spec = DipyReconInputSpec
    output_spec = DipyReconOutputSpec

    def _get_gtab(self, external_bvals=None, external_bvecs=None):
        small_delta = self.inputs.small_delta if isdefined(self.inputs.small_delta) else None
        big_delta = self.inputs.big_delta if isdefined(self.inputs.big_delta) else None
        bval_file = self.inputs.bval_file if external_bvals is None else external_bvals
        bvec_file = self.inputs.bvec_file if external_bvecs is None else external_bvecs
        gtab = gradient_table(
            bvals=np.loadtxt(bval_file),
            bvecs=np.loadtxt(bvec_file).T,
            b0_threshold=self.inputs.b0_threshold,
            big_delta=big_delta,
            small_delta=small_delta,
        )
        return gtab

    def _get_mask(self, amplitudes_img, gtab):
        if not isdefined(self.inputs.mask_file):
            dwi_data = amplitudes_img.get_fdata()
            LOGGER.warning("Creating an Otsu mask, check that the whole brain is covered.")
            _, mask_array = median_otsu(
                dwi_data, vol_idx=gtab.b0s_mask, median_radius=3, numpass=2
            )

            # Needed for synthetic data
            mask_array = mask_array * (dwi_data.sum(3) > 0)
            mask_img = nb.Nifti1Image(
                mask_array.astype("float32"), amplitudes_img.affine, amplitudes_img.header
            )
        else:
            mask_img = nb.load(self.inputs.mask_file)
            mask_array = mask_img.get_fdata() > 0
        return mask_img, mask_array

    def _save_scalar(self, data, suffix, runtime, ref_img):
        output_fname = fname_presuffix(self.inputs.dwi_file, suffix=suffix, newpath=runtime.cwd)
        nb.Nifti1Image(data, ref_img.affine, ref_img.header).to_filename(output_fname)
        return output_fname

    def _write_external_formats(self, runtime, fit_obj, mask_img, suffix):

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

    def _extrapolate_scheme(self, scheme_name, runtime, fit_obj, mask_img):
        if scheme_name not in ("ABCD", "HCP", "DSIQ5"):
            return
        output_dwi_file = fname_presuffix(
            self.inputs.dwi_file, suffix=scheme_name, newpath=runtime.cwd, use_ext=True
        )
        output_bval_file = fname_presuffix(
            self.inputs.dwi_file,
            suffix="{}.bval".format(scheme_name),
            newpath=runtime.cwd,
            use_ext=False,
        )
        output_bvec_file = fname_presuffix(
            self.inputs.dwi_file,
            suffix="{}.bvec".format(scheme_name),
            newpath=runtime.cwd,
            use_ext=False,
        )
        output_b_file = fname_presuffix(
            self.inputs.dwi_file,
            suffix="{}.b".format(scheme_name),
            newpath=runtime.cwd,
            use_ext=False,
        )
        # Copy in the bval and bvecs
        bval_file = pkgr("qsirecon", "data/schemes/{}.bval".format(scheme_name))
        bvec_file = pkgr("qsirecon", "data/schemes/{}.bvec".format(scheme_name))
        shutil.copyfile(bval_file, output_bval_file)
        shutil.copyfile(bvec_file, output_bvec_file)
        self._results["extrapolated_bvecs"] = bvec_file
        self._results["extrapolated_bvals"] = bval_file
        _convert_fsl_to_mrtrix(bval_file, bvec_file, output_b_file)
        self._results["extrapolated_b"] = output_b_file

        gtab_to_predict = self._get_gtab(external_bvals=bval_file, external_bvecs=bvec_file)
        new_data = fit_obj.predict(gtab_to_predict, S0=1000)
        # Clip negative values
        new_data = np.clip(new_data, 0, None)
        nb.Nifti1Image(new_data, mask_img.affine, mask_img.header).to_filename(output_dwi_file)
        self._results["extrapolated_dwi"] = output_dwi_file


class MAPMRIInputSpec(DipyReconInputSpec):
    radial_order = traits.Int(6, usedefault=True)
    laplacian_regularization = traits.Bool(True, usedefault=True)
    laplacian_weighting = traits.Either(traits.Str("GCV"), traits.Float(0.2), usedefault=True)
    positivity_constraint = traits.Bool(False, usedefault=True)
    pos_grid = traits.Int(15, usedefault=True)
    pos_radius = traits.Either(
        traits.Str("adaptive"), traits.Int(), default="adaptive", usedefault=True
    )
    anisotropic_scaling = traits.Bool(True, usedefault=True)
    eigenvalue_threshold = traits.Float(1e-04, usedefault=True)
    bval_threshold = traits.Float()
    dti_scale_estimation = traits.Bool(True, usedefault=True)
    static_diffusivity = traits.Float(0.7e-3, usedefault=True)
    cvxpy_solver = traits.Str()


class MAPMRIOutputSpec(DipyReconOutputSpec):
    rtop = File()
    lapnorm = File()
    msd = File()
    qiv = File()
    rtap = File()
    rtpp = File()
    ng = File()
    perng = File()
    parng = File()
    mapmri_coeffs = File()


class MAPMRIReconstruction(DipyReconInterface):
    input_spec = MAPMRIInputSpec
    output_spec = MAPMRIOutputSpec

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        dwi_img = nb.load(self.inputs.dwi_file)
        data = dwi_img.get_fdata(dtype="float32")
        mask_img, mask_array = self._get_mask(dwi_img, gtab)
        weighting = (
            "GCV" if self.inputs.laplacian_weighting == "GCV" else self.inputs.laplacian_weighting
        )

        if self.inputs.laplacian_regularization and self.inputs.positivity_constraint:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=True,
                laplacian_weighting=weighting,
                positivity_constraint=True,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling,
            )

        elif self.inputs.positivity_constraint:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=False,
                positivity_constraint=True,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling,
            )

        elif self.inputs.laplacian_regularization:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=True,
                laplacian_weighting=weighting,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling,
            )

        else:
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=self.inputs.radial_order,
                laplacian_regularization=False,
                positivity_constraint=False,
                bval_threshold=self.inputs.b0_threshold,
                anisotropic_scaling=self.inputs.anisotropic_scaling,
            )

        LOGGER.info("Fitting MAPMRI Model.")
        mapfit_aniso = map_model_aniso.fit(data, mask=mask_array)
        rtop = mapfit_aniso.rtop()
        self._results["rtop"] = self._save_scalar(rtop, "_rtop", runtime, dwi_img)

        ll = mapfit_aniso.norm_of_laplacian_signal()
        self._results["lapnorm"] = self._save_scalar(ll, "_lapnorm", runtime, dwi_img)

        m = mapfit_aniso.msd()
        self._results["msd"] = self._save_scalar(m, "_msd", runtime, dwi_img)

        q = mapfit_aniso.qiv()
        self._results["qiv"] = self._save_scalar(q, "_qiv", runtime, dwi_img)

        rtap = mapfit_aniso.rtap()
        self._results["rtap"] = self._save_scalar(rtap, "_rtap", runtime, dwi_img)

        rtpp = mapfit_aniso.rtpp()
        self._results["rtpp"] = self._save_scalar(rtpp, "_rtpp", runtime, dwi_img)

        coeffs = mapfit_aniso.mapmri_coeff
        self._results["mapmri_coeffs"] = self._save_scalar(coeffs, "_mapcoeffs", runtime, dwi_img)

        if self.inputs.anisotropic_scaling:
            ng = mapfit_aniso.ng()
            self._results["ng"] = self._save_scalar(ng, "_ng", runtime, dwi_img)

            perng = mapfit_aniso.ng_perpendicular()
            self._results["perng"] = self._save_scalar(perng, "_perng", runtime, dwi_img)

            parng = mapfit_aniso.ng_parallel()
            self._results["parng"] = self._save_scalar(parng, "_parng", runtime, dwi_img)

        # Write DSI Studio or MRtrix
        self._write_external_formats(runtime, mapfit_aniso, mask_img, "_MAPMRI")

        return runtime


class BrainSuiteShoreReconstructionInputSpec(DipyReconInputSpec):
    radial_order = traits.Int(6, usedefault=True)
    zeta = traits.Float(700, usedefault=True)
    tau = traits.Float(TAU_DEFAULT, usedefault=True)
    regularization = traits.Enum("L2", "L1", usedefault=True)
    # For L2
    lambdaN = traits.Float(1e-8, usedefault=True)
    lambdaL = traits.Float(1e-8, usedefault=True)
    # For L1
    regularization_weighting = traits.Str("CV", usedefault=True)
    l1_positive_constraint = traits.Bool(False, usedefault=True)
    l1_cv = traits.Either(traits.Int(3), None, usedefault=True)
    l1_maxiter = traits.Int(1000, usedefault=True)
    l1_verbose = traits.Bool(False, usedefault=True)
    l1_alpha = traits.Float(1.0, usedefault=True)
    # For EAP
    pos_grid = traits.Int(11, usedefault=True)
    pos_radius = traits.Float(20e-03, usedefault=True)


class BrainSuiteShoreReconstructionOutputSpec(DipyReconOutputSpec):
    shore_coeffs_image = File()
    rtop_image = File()
    alpha_image = File()
    r2_image = File()
    cnr_image = File()
    regularization_image = File()


class BrainSuiteShoreReconstruction(DipyReconInterface):
    input_spec = BrainSuiteShoreReconstructionInputSpec
    output_spec = BrainSuiteShoreReconstructionOutputSpec

    def _extrapolate_scheme(self, scheme_name, runtime, fit_obj, mask_array, mask_img):
        if scheme_name not in ("ABCD", "HCP", "DSIQ5"):
            return
        output_dwi_file = fname_presuffix(
            self.inputs.dwi_file, suffix=scheme_name, newpath=runtime.cwd, use_ext=True
        )
        output_bval_file = fname_presuffix(
            self.inputs.dwi_file,
            suffix="{}.bval".format(scheme_name),
            newpath=runtime.cwd,
            use_ext=False,
        )
        output_bvec_file = fname_presuffix(
            self.inputs.dwi_file,
            suffix="{}.bvec".format(scheme_name),
            newpath=runtime.cwd,
            use_ext=False,
        )
        output_b_file = fname_presuffix(
            self.inputs.dwi_file,
            suffix="{}.b".format(scheme_name),
            newpath=runtime.cwd,
            use_ext=False,
        )
        # Copy in the bval and bvecs
        bval_file = pkgr("qsirecon", "data/schemes/{}.bval".format(scheme_name))
        bvec_file = pkgr("qsirecon", "data/schemes/{}.bvec".format(scheme_name))
        shutil.copyfile(bval_file, output_bval_file)
        shutil.copyfile(bvec_file, output_bvec_file)
        self._results["extrapolated_bvecs"] = bvec_file
        self._results["extrapolated_bvals"] = bval_file
        _convert_fsl_to_mrtrix(bval_file, bvec_file, output_b_file)
        self._results["extrapolated_b"] = output_b_file

        prediction_gtab = self._get_gtab(external_bvals=bval_file, external_bvecs=bvec_file)
        prediction_shore = brainsuite_shore_basis(
            fit_obj.model.radial_order, fit_obj.model.zeta, prediction_gtab, fit_obj.model.tau
        )

        shore_array = fit_obj._shore_coef[mask_array]
        output_data = np.zeros(mask_array.shape + (len(prediction_gtab.bvals),))
        output_data[mask_array] = np.dot(shore_array, prediction_shore.T)
        # Clip negative values
        output_data = np.clip(output_data, 0, None)

        nb.Nifti1Image(output_data, mask_img.affine, mask_img.header).to_filename(output_dwi_file)
        self._results["extrapolated_dwi"] = output_dwi_file

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        b0s_mask = gtab.b0s_mask
        dwis_mask = np.logical_not(b0s_mask)
        dwi_img = nb.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata(dtype="float32")
        b0_images = dwi_data[..., b0s_mask]
        b0_mean = b0_images.mean(3)
        dwi_images = dwi_data[..., dwis_mask]

        mask_img, mask_array = self._get_mask(dwi_img, gtab)

        final_bvals = np.concatenate([np.array([0]), gtab.bvals[dwis_mask]])
        final_bvecs = np.row_stack([np.array([0.0, 0.0, 0.0]), gtab.bvecs[dwis_mask]])
        final_data = np.concatenate([b0_mean[..., np.newaxis], dwi_images], 3)
        final_grads = gradient_table(
            bvals=final_bvals,
            bvecs=final_bvecs,
            b0_threshold=50,
            big_delta=self.inputs.big_delta,
            small_delta=self.inputs.small_delta,
        )

        # Cleanup
        del dwi_images
        del b0_images
        del dwi_data

        bss_model = BrainSuiteShoreModel(
            final_grads,
            regularization=self.inputs.regularization,
            radial_order=self.inputs.radial_order,
            zeta=self.inputs.zeta,
            tau=self.inputs.tau,
            # For L2
            lambdaN=self.inputs.lambdaN,
            lambdaL=self.inputs.lambdaL,
            # For L1
            regularization_weighting=self.inputs.regularization_weighting,
            l1_positive_constraint=self.inputs.l1_positive_constraint,
            l1_cv=self.inputs.l1_cv,
            l1_maxiter=self.inputs.l1_maxiter,
            l1_verbose=self.inputs.l1_verbose,
            l1_alpha=self.inputs.l1_alpha,
            # For EAP
            pos_grid=self.inputs.pos_grid,
        )
        bss_fit = bss_model.fit(final_data, mask=mask_array)
        rtop = bss_fit.rtop_signal()
        coeffs = bss_fit.shore_coeff

        coeffs_file = fname_presuffix(
            self.inputs.dwi_file, suffix="_shore_coeff", newpath=runtime.cwd
        )
        rtop_file = fname_presuffix(self.inputs.dwi_file, suffix="_rtop", newpath=runtime.cwd)
        alphas_file = fname_presuffix(self.inputs.dwi_file, suffix="_alpha", newpath=runtime.cwd)
        r2_file = fname_presuffix(self.inputs.dwi_file, suffix="_r2", newpath=runtime.cwd)
        cnr_file = fname_presuffix(self.inputs.dwi_file, suffix="_cnr", newpath=runtime.cwd)
        regl_file = fname_presuffix(
            self.inputs.dwi_file, suffix="_regularization", newpath=runtime.cwd
        )

        nb.Nifti1Image(coeffs, dwi_img.affine, dwi_img.header).to_filename(coeffs_file)
        nb.Nifti1Image(rtop, dwi_img.affine, dwi_img.header).to_filename(rtop_file)
        nb.Nifti1Image(bss_fit.regularization, dwi_img.affine, dwi_img.header).to_filename(
            regl_file
        )
        nb.Nifti1Image(bss_fit.r2, dwi_img.affine, dwi_img.header).to_filename(r2_file)
        nb.Nifti1Image(bss_fit.cnr, dwi_img.affine, dwi_img.header).to_filename(cnr_file)
        nb.Nifti1Image(bss_fit.alpha, dwi_img.affine, dwi_img.header).to_filename(alphas_file)
        self._results["shore_coeffs_image"] = coeffs_file
        self._results["rtop_image"] = rtop_file
        self._results["alpha_image"] = alphas_file
        self._results["r2_image"] = r2_file
        self._results["cnr_image"] = cnr_file
        self._results["regularization_image"] = regl_file

        # Write DSI Studio or MRtrix
        self._write_external_formats(runtime, bss_fit, mask_img, "_BS3dSHORE")
        # Make HARDIs if desired
        extrapolate = self.inputs.extrapolate_scheme
        if isdefined(extrapolate):
            self._extrapolate_scheme(extrapolate, runtime, bss_fit, mask_array, mask_img)
        return runtime


class TensorReconstructionInputSpec(DipyReconInputSpec):
    pass


class TensorReconstructionOutputSpec(DipyReconOutputSpec):
    color_fa_image = File()
    fa_image = File()
    md_image = File()
    rd_image = File()
    ad_image = File()
    cnr_image = File()


class TensorReconstruction(DipyReconInterface):
    input_spec = TensorReconstructionInputSpec
    output_spec = TensorReconstructionOutputSpec

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        dwi_img = nb.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata(dtype="float32")
        mask_img, mask_array = self._get_mask(dwi_img, gtab)

        # Fit it
        tenmodel = dti.TensorModel(gtab)
        ten_fit = tenmodel.fit(dwi_data, mask_array)
        lower_triangular = ten_fit.lower_triangular()
        tensor_img = nifti1_symmat(lower_triangular, dwi_img.affine)
        output_tensor_file = fname_presuffix(
            self.inputs.dwi_file, suffix="tensor", newpath=runtime.cwd, use_ext=True
        )
        tensor_img.to_filename(output_tensor_file)

        # FA MD RD and AD
        for metric in ["fa", "md", "rd", "ad", "color_fa"]:
            data = getattr(ten_fit, metric).astype("float32")
            out_name = fname_presuffix(
                self.inputs.dwi_file, suffix=metric, newpath=runtime.cwd, use_ext=True
            )
            nb.Nifti1Image(data, dwi_img.affine).to_filename(out_name)
            self._results[metric + "_image"] = out_name

        return runtime


class _KurtosisReconstructionInputSpec(DipyReconInputSpec):
    kurtosis_clip_min = traits.Float(-0.42857142857142855, usedefault=True)
    kurtosis_clip_max = traits.Float(10.0, usedefault=True)


class _KurtosisReconstructionOutputSpec(DipyReconOutputSpec):
    tensor = File()
    ad = File()
    ak = File()
    colorFA = File()
    fa = File()
    kfa = File()
    linearity = File()
    md = File()
    mk = File()
    mkt = File()
    planarity = File()
    rd = File()
    rk = File()
    sphericity = File()


class KurtosisReconstruction(DipyReconInterface):
    input_spec = _KurtosisReconstructionInputSpec
    output_spec = _KurtosisReconstructionOutputSpec

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        dwi_img = nb.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata(dtype="float32")
        mask_img, mask_array = self._get_mask(dwi_img, gtab)

        # Fit it
        dkimodel = dki.DiffusionKurtosisModel(gtab)
        dkifit = dkimodel.fit(dwi_data, mask_array)
        lower_triangular = dkifit.lower_triangular()
        tensor_img = nifti1_symmat(lower_triangular, dwi_img.affine)
        output_tensor_file = fname_presuffix(
            self.inputs.dwi_file, suffix="DKItensor", newpath=runtime.cwd, use_ext=True
        )
        tensor_img.to_filename(output_tensor_file)
        self._results["tensor"] = output_tensor_file

        # FA MD RD and AD
        metric_attrs = {
            "colorFA": "color_fa",
        }
        base_metrics = [
            "ad",
            "colorFA",
            "fa",
            "kfa",
            "linearity",
            "md",
            "planarity",
            "rd",
            "sphericity",
        ]
        for metric in base_metrics:
            metric_attr = metric_attrs.get(metric, metric)
            data = np.nan_to_num(getattr(dkifit, metric_attr).astype("float32"), 0)
            out_name = fname_presuffix(
                self.inputs.dwi_file, suffix="DKI" + metric, newpath=runtime.cwd, use_ext=True
            )
            nb.Nifti1Image(data, dwi_img.affine).to_filename(out_name)
            self._results[metric] = out_name

        # Get the kurtosis metrics
        kurtosis_metrics = ["ak", "mk", "mkt", "rk"]
        for metric in kurtosis_metrics:
            data = np.nan_to_num(
                getattr(dkifit, metric)(
                    float(self.inputs.kurtosis_clip_min), float(self.inputs.kurtosis_clip_max)
                ),
                0,
            )
            out_name = fname_presuffix(
                self.inputs.dwi_file, suffix="DKI" + metric, newpath=runtime.cwd, use_ext=True
            )
            nb.Nifti1Image(data, dwi_img.affine).to_filename(out_name)
            self._results[metric] = out_name

        return runtime


class _KurtosisReconstructionMicrostructureInputSpec(DipyReconInputSpec):
    kurtosis_clip_min = traits.Float(-0.42857142857142855, usedefault=True)
    kurtosis_clip_max = traits.Float(10.0, usedefault=True)


class _KurtosisReconstructionMicrostructureOutputSpec(DipyReconOutputSpec):
    ad = File()
    ade = File()
    awf = File()
    axonald = File()
    kfa = File()
    md = File()
    rd = File()
    rde = File()
    tortuosity = File()
    trace = File()


class KurtosisReconstructionMicrostructure(DipyReconInterface):
    input_spec = _KurtosisReconstructionMicrostructureInputSpec
    output_spec = _KurtosisReconstructionMicrostructureOutputSpec

    def _run_interface(self, runtime):
        gtab = self._get_gtab()
        dwi_img = nb.load(self.inputs.dwi_file)
        dwi_data = dwi_img.get_fdata(dtype="float32")
        mask_img, mask_array = self._get_mask(dwi_img, gtab)

        # Fit it
        dkimodel = dki_micro.KurtosisMicrostructureModel(gtab)
        dkifit = dkimodel.fit(dwi_data, mask_array)

        # FA MD RD and AD
        metric_attrs = {
            "ade": "hindered_ad",
            "rde": "hindered_rd",
            "axonald": "axonal_diffusivity",
        }
        base_metrics = [
            "ad",
            "ade",
            "awf",
            "axonald",
            "kfa",
            "md",
            "rd",
            "rde",
            "tortuosity",
            "trace",
        ]
        for metric in base_metrics:
            metric_attr = metric_attrs.get(metric, metric)
            data = np.nan_to_num(getattr(dkifit, metric_attr).astype("float32"), 0)
            out_name = fname_presuffix(
                self.inputs.dwi_file,
                suffix="DKIMicro" + metric,
                newpath=runtime.cwd,
                use_ext=True,
            )
            nb.Nifti1Image(data, dwi_img.affine).to_filename(out_name)
            self._results[metric] = out_name

        return runtime
