#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op
import zipfile
from copy import deepcopy

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import numpy as np
import pandas as pd
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.mrtrix3 import Generate5tt, MRConvert, ResponseSD
from nipype.interfaces.mrtrix3.base import MRTrix3Base, MRTrix3BaseInputSpec
from nipype.interfaces.mrtrix3.preprocess import ResponseSDInputSpec
from nipype.interfaces.mrtrix3.tracking import Tractography, TractographyInputSpec
from nipype.interfaces.mrtrix3.utils import Generate5ttInputSpec
from nipype.utils.filemanip import fname_presuffix, split_filename, which
from scipy.io.matlab import loadmat, savemat

LOGGER = logging.getLogger("nipype.interface")
RC3_ROOT = which("average_response")  # Only exists in RC3
if RC3_ROOT is not None:
    # Use the directory containing average_response
    RC3_ROOT = os.path.split(RC3_ROOT)[0]

_SS3T_EXE = which("ss3t_csd_beta1")
if _SS3T_EXE is None:
    if os.getenv("SS3T_HOME"):
        SS3T_ROOT = os.getenv("SS3T_HOME")
    else:
        if not os.path.exists("/opt/3Tissue/bin/ss3t_csd_beta1"):
            LOGGER.warn("Check installation of 3Tissue")
        SS3T_ROOT = "/opt/3Tissue/bin"
else:
    SS3T_ROOT = os.path.split(_SS3T_EXE)[0]


class TckGenInputSpec(TractographyInputSpec):
    power = traits.CFloat(argstr="-power %f")
    select = traits.CInt(argstr="-select %d")
    select = traits.CInt(
        argstr="-select %d",
        desc=(
            "set the desired number of tracks. The program will continue"
            " to generate tracks until this number of tracks have been "
            "selected and written to the output file"
        ),
    )
    n_tracks = traits.Int(desc="NOT supported, do not use")
    quiet = traits.Bool(argstr="-quiet")


class TckGen(Tractography):
    input_spec = TckGenInputSpec


class MRTrixGradientTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)


class MRTrixGradientTableOutputSpec(TraitedSpec):
    gradient_file = File(exists=True)


class MRTrixGradientTable(SimpleInterface):
    input_spec = MRTrixGradientTableInputSpec
    output_spec = MRTrixGradientTableOutputSpec

    def _run_interface(self, runtime):
        gtab_fname = fname_presuffix(
            self.inputs.bval_file, suffix=".b", newpath=runtime.cwd, use_ext=False
        )
        _convert_fsl_to_mrtrix(self.inputs.bval_file, self.inputs.bvec_file, gtab_fname)
        self._results["gradient_file"] = gtab_fname
        return runtime


def _convert_fsl_to_mrtrix(bval_file, bvec_file, output_fname):
    vecs = np.loadtxt(bvec_file)
    vals = np.loadtxt(bval_file)
    gtab = np.column_stack([vecs.T, vals]) * np.array([-1, -1, 1, 1])
    np.savetxt(output_fname, gtab, fmt=["%.8f", "%.8f", "%.8f", "%d"])


class MRTrixIngressInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    suffix = traits.Str("", usedefault=True)


class MRTrixIngressOutputSpec(TraitedSpec):
    mif_file = File()


class MRTrixIngress(SimpleInterface):
    input_spec = MRTrixIngressInputSpec
    output_spec = MRTrixIngressOutputSpec

    def _run_interface(self, runtime):
        output_mif = fname_presuffix(
            self.inputs.dwi_file,
            suffix=self.inputs.suffix + ".mif",
            newpath=runtime.cwd,
            use_ext=False,
        )
        if isdefined(self.inputs.b_file):
            convert = MRConvert(
                in_file=self.inputs.dwi_file, grad_file=self.inputs.b_file, out_file=output_mif
            )
        elif isdefined(self.inputs.bval_file) and isdefined(self.inputs.bvec_file):
            convert = MRConvert(
                in_file=self.inputs.dwi_file,
                in_bval=self.inputs.bval_file,
                in_bvec=self.inputs.bvec_file,
                out_file=output_mif,
            )
        else:
            raise Exception("No valid mrtrix gradient files or fsl bval/bvec files specified")
        convert_run = convert.run()
        self._results["mif_file"] = convert_run.outputs.out_file

        return runtime


class GenerateMasked5ttInputSpec(Generate5ttInputSpec):
    algorithm = traits.Enum(
        "fsl",
        "gif",
        "freesurfer",
        "hsvs",
        argstr="%s",
        position=0,
        mandatory=True,
        desc="tissue segmentation algorithm",
    )
    in_file = traits.Either(
        File(exists=True),
        traits.Directory(exists=True),
        argstr="%s",
        mandatory=True,
        position=1,
        desc="input T1w image or FreeSurfer directory",
    )
    out_file = File(argstr="%s", genfile=True, position=2, desc="output image")
    mask = File(exists=True, argstr="-mask %s")
    amygdala_hipppocampi_subcortical_gm = traits.Bool(argstr="-sgm_amyg_hipp")
    white_stem = traits.Bool(argstr="-white_stem")
    thalami_method = traits.Enum("nuclei", "first", "aseg", argstr="-thalami %s")
    hippocampi_method = traits.Enum("subfields", "first", "aseg", argstr="-hippocampi %s")


class GenerateMasked5tt(Generate5tt):
    input_spec = GenerateMasked5ttInputSpec

    def _gen_filename(self, name):
        if name == "out_file":
            output = self.inputs.out_file
            if not isdefined(output):
                _, fname, _ = split_filename(self.inputs.in_file)
                output = fname + "_5tt.mif"
            return output
        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = op.abspath(self._gen_filename("out_file"))
        return outputs


class Dwi2ResponseInputSpec(ResponseSDInputSpec):
    wm_file = File(argstr="%s", position=-3, genfile=True, desc="output WM response text file")
    gm_file = File(argstr="%s", genfile=True, position=-2, desc="output GM response text file")
    csf_file = File(argstr="%s", position=-1, genfile=True, desc="output CSF response text file")
    max_sh = InputMultiObject(
        traits.Int,
        argstr="-lmax %s",
        sep=",",
        desc=(
            "maximum harmonic degree of response function - single value for "
            "single-shell response, list for multi-shell response"
        ),
    )


class Dwi2Response(ResponseSD):
    input_spec = Dwi2ResponseInputSpec

    def _format_arg(self, name, spec, val):
        if self.inputs.algorithm not in ("dhollander", "msmt_5tt"):
            if name in ("gm_file", "csf_file"):
                return ""
        return super(Dwi2Response, self)._format_arg(name, spec, val)

    def _gen_filename(self, name):
        if name in ("gm_file", "csf_file", "wm_file"):
            output = getattr(self.inputs, name)
            if not isdefined(output):
                _, fname, ext = split_filename(self.inputs.in_file)
                output = fname + "_" + name.split("_")[0] + ".txt"
            return output
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["wm_file"] = op.abspath(self._gen_filename("wm_file"))
        if self.inputs.algorithm in ("dhollander", "msmt_5tt"):
            outputs["gm_file"] = op.abspath(self._gen_filename("gm_file"))
            outputs["csf_file"] = op.abspath(self._gen_filename("csf_file"))
        return outputs


class SS3TBase(MRTrix3Base):

    def _pre_run_hook(self, runtime):
        """Sets the PATH to contain 3Tissue instead of RC3."""

        # If 3Tissue is the only MRtrix, there will be no path to average_response
        if RC3_ROOT is None:
            return runtime

        # Replace the RC3 mrtrix with 3Tissue in PATH
        old_path = runtime.environ.get("PATH")
        new_path = old_path.replace(RC3_ROOT, SS3T_ROOT)
        runtime.environ["PATH"] = new_path
        return runtime


class SS3TDwi2Response(SS3TBase, Dwi2Response):
    pass


class MTNormalizeInputSpec(MRTrix3BaseInputSpec):
    wm_odf = File(argstr="%s", position=0, mandatory=True, desc="WM ODF image")
    wm_normed_odf = File(
        argstr="%s",
        position=1,
        name_template="%s_mtnorm",
        keep_extension=True,
        name_source="wm_odf",
        desc="output WM normed_odf",
    )
    gm_odf = File(argstr="%s", position=2, desc="GM ODF image")
    gm_normed_odf = File(
        argstr="%s",
        position=3,
        name_template="%s_mtnorm",
        keep_extension=True,
        name_source="gm_odf",
        desc="output GM normed_odf",
    )
    csf_odf = File(argstr="%s", position=4, desc="CSF ODF image")
    csf_normed_odf = File(
        argstr="%s",
        position=5,
        name_template="%s_mtnorm",
        keep_extension=True,
        name_source="csf_odf",
        desc="output CSF normed_odf",
    )
    mask_file = File(exists=True, mandatory=True, argstr="-mask %s", desc="mask image")
    inlier_mask = File(
        argstr="-check_mask %s",
        name_template="%s_inlier_mask.nii.gz",
        keep_extension=False,
        name_source="wm_odf",
        desc="estimated spatially varying intensity level that is used for normalisation",
    )
    norm_image = File(
        argstr="-check_norm %s",
        name_template="%s_norm_image.nii.gz",
        keep_extension=False,
        name_source="wm_odf",
        desc="final mask used to compute the normalisation. This mask"
        " excludes regions identified as outliers by the optimisation process.",
    )


class MTNormalizeOutputSpec(TraitedSpec):
    wm_normed_odf = File(desc="normalized WM ODF")
    gm_normed_odf = File(desc="normalized GM ODF")
    csf_normed_odf = File(desc="normalized CSF ODF")
    norm_image = File(
        desc="estimated spatially varying intensity level that is used " "for normalisation"
    )
    inlier_mask = File(
        desc="final mask used to compute the normalisation. This mask"
        " excludes regions identified as outliers by the optimisation process."
    )


class MTNormalize(SS3TBase):
    _cmd = "mtnormalise"
    input_spec = MTNormalizeInputSpec
    output_spec = MTNormalizeOutputSpec

    def _gen_filename(self, name):
        _, fname, ext = split_filename(self.inputs.in_file)
        if name.endswith("_norm_odf"):
            tissue_type = name.split("_")[0]
            output = getattr(self.inputs, tissue_type + "_odf")
            if not isdefined(output):
                output = fname + "_" + tissue_type + "_normed" + ext
            return output
        if name == "norm_image":
            return fname + "_mtbias.nii.gz"
        if name == "inlier_mask":
            return fname + "_mtmask.nii.gz"
        return None


class EstimateFODInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum(
        "csd", "msmt_csd", argstr="%s", position=-8, mandatory=True, desc="FOD algorithm"
    )
    in_file = File(exists=True, argstr="%s", position=-7, mandatory=True, desc="input DWI image")
    wm_txt = File(argstr="%s", position=-6, mandatory=True, desc="WM response text file")
    wm_odf = File(argstr="%s", position=-5, genfile=True, desc="output WM ODF")
    gm_txt = File(argstr="%s", position=-4, requires=["csf_txt"], desc="GM response text file")
    gm_odf = File(
        argstr="%s", position=-3, genfile=True, requires=["gm_txt"], desc="output GM ODF"
    )
    csf_txt = File(argstr="%s", position=-2, desc="CSF response text file")
    csf_odf = File(
        argstr="%s", position=-1, genfile=True, requires=["csf_txt"], desc="output CSF ODF"
    )
    mask_file = File(exists=True, argstr="-mask %s", desc="mask image")
    shell = traits.List(
        traits.Float, sep=",", argstr="-shell %s", desc="specify one or more dw gradient shells"
    )
    max_sh = InputMultiObject(
        traits.Int,
        argstr="-lmax %s",
        sep=",",
        desc="maximum harmonic degree of response function - single value for single-shell "
        "response, list for multi-shell response",
    )
    in_dirs = File(
        exists=True,
        argstr="-directions %s",
        desc=(
            "specify the directions over which to apply the non-negativity "
            "constraint (by default, the built-in 300 direction set is "
            "used). These should be supplied as a text file containing the "
            "[ az el ] pairs for the directions."
        ),
    )


class EstimateFODOutputSpec(TraitedSpec):
    wm_odf = File(desc="output WM ODF")
    gm_odf = File(desc="output GM ODF")
    csf_odf = File(desc="output CSF ODF")


class EstimateFOD(MRTrix3Base):
    """
    Estimate fibre orientation distributions from diffusion data using spherical deconvolution
    """

    _cmd = "dwi2fod"
    input_spec = EstimateFODInputSpec
    output_spec = EstimateFODOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["wm_odf"] = op.abspath(self._gen_filename("wm_odf"))
        if self.inputs.algorithm in ("msmt_csd", "ss3t"):
            outputs["gm_odf"] = op.abspath(self._gen_filename("gm_odf"))
            outputs["csf_odf"] = op.abspath(self._gen_filename("csf_odf"))
        return outputs

    def _format_arg(self, name, spec, value):
        if self.inputs.algorithm == "csd":
            if name in ("gm_odf", "gm_txt", "csf_odf", "csf_txt"):
                return ""
        return super(EstimateFOD, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name in ("gm_odf", "gm_txt", "wm_odf", "wm_txt", "csf_odf", "csf_txt"):
            output = getattr(self.inputs, name)
            if not isdefined(output):
                _, fname, _ = split_filename(self.inputs.in_file)
                ext = ".txt" if name.endswith("txt") else ".mif"
                output = fname + "_" + name.split("_")[0] + ext
            return output
        return None


class SS3TEstimateFODInputSpec(EstimateFODInputSpec):
    algorithm = traits.Str("ss3t", desc="Not needed for ss3t")


class SS3TEstimateFOD(SS3TBase, EstimateFOD):
    _cmd = "ss3t_csd_beta1" if SS3T_ROOT is None else op.join(SS3T_ROOT, "ss3t_csd_beta1")
    input_spec = SS3TEstimateFODInputSpec
    output_spec = EstimateFODOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["wm_odf"] = op.abspath(self._gen_filename("wm_odf"))
        outputs["gm_odf"] = op.abspath(self._gen_filename("gm_odf"))
        outputs["csf_odf"] = op.abspath(self._gen_filename("csf_odf"))
        return outputs

    def _gen_filename(self, name):
        if name in ("gm_odf", "gm_txt", "wm_odf", "wm_txt", "csf_odf", "csf_txt"):
            output = getattr(self.inputs, name)
            if not isdefined(output):
                _, fname, _ = split_filename(self.inputs.in_file)
                ext = ".txt" if name.endswith("txt") else ".mif"
                output = fname + "_" + name.split("_")[0] + ext
            return output
        return None


class SIFT2InputSpec(MRTrix3BaseInputSpec):
    in_tracks = File(argstr="%s", exists=True, mandatory=True, position=-3, desc="input tck file")
    in_fod = File(argstr="%s", position=-2, exists=True, mandatory=True, desc="input FOD SH file")
    out_weights = File(
        argstr="%s",
        position=-1,
        genfile=True,
        desc="output text file containing the weighting" "factor for each streamline",
    )
    act_file = File(
        exists=True,
        argstr="-act %s",
        desc=(
            "use the Anatomically-Constrained Tractography framework during"
            " tracking; provided image must be in the 5TT "
            "(five - tissue - type) format"
        ),
    )
    fd_scale_gm = traits.Bool(
        requires=["act_file"],
        argstr="-fd_scale_gm",
        desc="provide this option "
        "(in conjunction with -act) to heuristically downsize the fibre density estimates "
        "based on the presence of GM in the voxel. This can assist in reducing tissue interface "
        "effects when using a single-tissue deconvolution algorithm",
    )
    no_dilate_lut = traits.Bool(
        argstr="-no_dilate_lut",
        desc="do NOT dilate FOD lobe lookup tables; only map "
        "streamlines to FOD lobes if the precise tangent lies within the angular spread of "
        "that lobe",
    )
    make_null_lobes = traits.Bool(
        argstr="-make_null_lobes",
        desc="add an additional FOD lobe to each voxel, with zero "
        "integral, that covers all directions with zero / negative FOD amplitudes",
    )
    remove_untracked = traits.Bool(
        argstr="-remove_untracked",
        desc="remove FOD lobes that do not have any streamline "
        "density attributed to them; this improves filtering slightly, at the expense of longer "
        "computation time (and you can no longer do quantitative comparisons between "
        "reconstructions if this is enabled)",
    )
    fd_thresh = traits.Float(
        argstr="-fd_thresh %f",
        desc="fibre density threshold; exclude an FOD lobe from "
        "filtering processing if its integral is less than this amount (streamlines will still "
        "be mapped to it, but it will not contribute to the cost function or the filtering)",
    )
    out_mu = traits.File(
        argstr="-out_mu %s",
        genfile=True,
        desc="output the final value of SIFT proportionality " "coefficient mu to a text file",
    )


class SIFT2OutputSpec(TraitedSpec):
    out_mu = File(exists=True)
    out_weights = File(exists=True)


class SIFT2(MRTrix3Base):
    input_spec = SIFT2InputSpec
    output_spec = SIFT2OutputSpec
    _cmd = "tcksift2"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_mu"] = op.abspath(self._gen_filename("out_mu"))
        outputs["out_weights"] = op.abspath(self._gen_filename("out_weights"))
        return outputs

    def _gen_filename(self, name):
        _, fname, _ = split_filename(self.inputs.in_fod)
        output = getattr(self.inputs, name)
        if name == "out_mu":
            if not isdefined(output):
                output = fname + "_mu.txt"
            return output
        if name == "out_weights":
            if not isdefined(output):
                output = fname + "_weights.csv"
            return output
        return None


class GlobalTractographyInputSpec(MRTrix3BaseInputSpec):
    dwi_file = File(
        argstr="%s", exists=True, mandatory=True, position=-3, desc="full dwi file (source)"
    )
    wm_txt = File(
        exists=True,
        argstr="%s",
        mandatory=True,
        position=-2,
        desc="wm response function (response)",
    )
    mask = File(
        exists=True,
        argstr="-mask %s",
        desc="only reconstruct the tractogram within the specified brain mask image.",
    )
    out_tracks = File(
        argstr="%s", position=-1, genfile=True, desc="the globally-optimized streamlines (tracks)"
    )
    gm_txt = File(argstr="-riso %s", exists=True, desc="gm isotropic response functions")
    csf_txt = File(argstr="-riso %s", exists=True, desc="csf isotropic response functions")
    out_fod = File(
        argstr="-fod %s",
        genfile=True,
        desc="Predicted fibre orientation distribution function (fODF).This fODF is "
        "estimated as part of the global track optimization, and therefore incorporates "
        "the spatial regularization that it imposes. Internally, the fODF is "
        "represented as a discrete sum of apodized point spread functions (aPSF) "
        "oriented along the directions of all particles in the voxel, used to predict "
        "the DWI signal from the particle configuration",
    )
    out_isotropic_fraction = File(
        argstr="-fiso %s",
        requires=["csf_txt"],
        genfile=True,
        desc=" Predicted isotropic fractions of the tissues for which response "
        "functions were provided with isotropic_response_txt. Typically, "
        "these are CSF and GM.",
    )
    niter = traits.Int(
        1e9,
        argstr="-niter %d",
        desc="the number of iterations of the metropolis hastings optimizer. " "(default = 10M)",
    )
    out_residual_energy = File(
        argstr="-eext %s", genfile=True, desc=" Residual external energy in every voxel."
    )


class GlobalTractographyOutputSpec(TraitedSpec):
    wm_odf = File(
        exists=True,
        desc="Predicted fibre orientation distribution function (fODF).This fODF is "
        "estimated as part of the global track optimization, and therefore incorporates "
        "the spatial regularization that it imposes. Internally, the fODF is represented "
        "as a discrete sum of apodized point spread functions (aPSF) oriented along the "
        "directions of all particles in the voxel, used to predict the DWI signal from the "
        "particle configuration.",
    )
    isotropic_fraction = File(
        exists=True,
        desc="Predicted isotropic fractions of the tissues for "
        "which response functions were provided with -riso. Typically, "
        "these are CSF and GM.",
    )
    residual_energy = File(exists=True, desc="Residual external energy in every voxel")
    tck_file = File(exists=True, desc="global tck file")


class GlobalTractography(MRTrix3Base):
    input_spec = GlobalTractographyInputSpec
    output_spec = GlobalTractographyOutputSpec
    _cmd = "tckglobal"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["wm_odf"] = op.abspath(self._gen_filename("out_fod"))
        outputs["isotropic_fraction"] = op.abspath(self._gen_filename("out_isotropic_fraction"))
        outputs["tck_file"] = op.abspath(self._gen_filename("out_tracks"))
        outputs["residual_energy"] = op.abspath(self._gen_filename("out_residual_energy"))
        return outputs

    def _gen_filename(self, name):
        output = getattr(self.inputs, name)
        _, fname, _ = split_filename(self.inputs.dwi_file)
        if name == "out_isotropic_fraction":
            if not isdefined(output):
                output = fname + "_tckglobalISOfraction.mif"
            return output
        if name == "out_fod":
            if not isdefined(output):
                output = fname + "_tckglobalFOD.mif"
            return output
        if name == "out_tracks":
            if not isdefined(output):
                output = fname + "_tckglobal.tck"
            return output
        if name == "out_residual_energy":
            if not isdefined(output):
                output = fname + "_residualEnergy.mif"
            return output
        return None


class BuildConnectomeInputSpec(CommandLineInputSpec):
    atlas_name = traits.Str(desc="name of atlas (for variables in matfile)")
    atlas_labels_file = File(exists=True, desc="parcellation labels file")
    measure = traits.Str(desc="Name of the connectivity measure")
    in_file = File(
        exists=True, argstr="%s", mandatory=True, position=-3, desc="input tractography"
    )
    in_parc = File(exists=True, argstr="%s", position=-2, desc="parcellation file")
    out_file = File(
        "connectome.csv",
        argstr="%s",
        mandatory=True,
        position=-1,
        usedefault=True,
        desc="output file after processing",
    )
    out_assignments = File(argstr="-out_assignments %s", desc="file with streamline assignments")
    nthreads = traits.Int(
        argstr="-nthreads %d",
        desc="number of threads. if zero, the number" " of available cpus will be used",
        nohash=True,
    )
    vox_lookup = traits.Bool(
        argstr="-assignment_voxel_lookup",
        desc="use a simple voxel lookup value at each streamline endpoint",
    )
    search_radius = traits.Float(
        argstr="-assignment_radial_search %f",
        desc="perform a radial search from each streamline endpoint to locate "
        "the nearest node. Argument is the maximum radius in mm; if no node is"
        " found within this radius, the streamline endpoint is not assigned to"
        " any node.",
    )
    search_reverse = traits.Float(
        argstr="-assignment_reverse_search %f",
        desc="traverse from each streamline endpoint inwards along the "
        "streamline, in search of the last node traversed by the streamline. "
        "Argument is the maximum traversal length in mm (set to 0 to allow "
        "search to continue to the streamline midpoint).",
    )
    search_forward = traits.Float(
        argstr="-assignment_forward_search %f",
        desc="project the streamline forwards from the endpoint in search of a"
        "parcellation node voxel. Argument is the maximum traversal length in "
        "mm.",
    )
    stat_edge = traits.Enum("sum", "mean", "min", "max", argstr="-stat_edge %s", usedefault=True)
    length_scale = traits.Enum("None", "length", "invlength", argstr="%s")
    scale_invnodevol = traits.Bool(False, argstr="-scale_invnodevol")
    in_scalar = File(
        exists=True,
        argstr="-scale_file %s",
        desc="provide the associated image " "for the mean_scalar metric",
    )
    use_sift_weights = traits.Bool(default=False, usedefault=True)
    in_weights = File(
        exists=True,
        argstr="-tck_weights_in %s",
        desc="specify a text scalar " "file containing the streamline weights",
    )
    keep_unassigned = traits.Bool(
        argstr="-keep_unassigned",
        desc="By default, the program discards the"
        " information regarding those streamlines that are not successfully "
        "assigned to a node pair. Set this option to keep these values (will "
        "be the first row/column in the output matrix)",
    )
    zero_diagonal = traits.Bool(
        argstr="-zero_diagonal",
        desc="set all diagonal entries in the matrix "
        "to zero (these represent streamlines that connect to the same node at"
        " both ends)",
    )
    symmetric = traits.Bool(argstr="-symmetric", desc="Make matrices symmetric on output")
    quiet = traits.Bool(argstr="-quiet")


class BuildConnectomeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output connectivity csv")
    connectivity_matfile = File(exists=True, desc="the matfile containing connectivity data")
    out_assignments = File(exists=True, desc="streamline assignment csv")


class BuildConnectome(MRTrix3Base):
    """Generate a connectome matrix from a streamlines file and a node parcellation image.

    Example
    -------
    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mat = mrt.BuildConnectome()
    >>> mat.inputs.in_file = "tracks.tck"
    >>> mat.inputs.in_parc = "aparc+aseg.nii"
    >>> mat.cmdline                               # doctest: +ELLIPSIS
    "tck2connectome tracks.tck aparc+aseg.nii connectome.csv"
    >>> mat.run()                                 # doctest: +SKIP
    """

    _cmd = "tck2connectome"
    input_spec = BuildConnectomeInputSpec
    output_spec = BuildConnectomeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)
        prefix = f"{self.inputs.atlas_name}_{self.inputs.measure}"
        outputs["connectivity_matfile"] = op.abspath(f"{prefix}_connectivity.mat")
        if isdefined(self.inputs.out_assignments):
            outputs["out_assignments"] = op.abspath(self.inputs.out_assignments)
        return outputs

    def _post_run_hook(self, runtime):
        atlas_name = self.inputs.atlas_name
        atlas_labels_df = pd.read_table(self.inputs.atlas_labels_file)

        atlas_labels_df["index"] = atlas_labels_df["index"].astype(int)
        if 0 in atlas_labels_df["index"].values:
            print(f"WARNING: Atlas {atlas_name} has a 0 index. This index will be dropped.")
            atlas_labels_df = atlas_labels_df.loc[atlas_labels_df["index"] != 0]

        # Aggregate the connectivity/network data from DSI Studio
        connectivity_data = {
            f"atlas_{atlas_name}_region_ids": atlas_labels_df["index"].values,
            f"atlas_{atlas_name}_region_labels": atlas_labels_df["label"].values,
        }

        # get the connectivity matrix
        prefix = f"{atlas_name}_{self.inputs.measure}"
        connectivity_data[f"atlas_{prefix}_connectivity"] = np.loadtxt(
            self.inputs.out_file, delimiter=","
        )
        connectivity_data["command"] = self.cmdline
        merged_matfile = op.join(runtime.cwd, f"{prefix}_connectivity.mat")
        savemat(merged_matfile, connectivity_data, long_field_names=True)
        return runtime

    def _format_arg(self, name, spec, val):
        if name == "length_scale":
            if val == "length":
                return "-scale_length"
            if val == "invlength":
                return "-scale_invlength"
            return ""
        if name == "in_weights":
            if self.inputs.use_sift_weights:
                return spec.argstr % val
            return ""
        return super(BuildConnectome, self)._format_arg(name, spec, val)


class MRTrixAtlasGraphInputSpec(BuildConnectomeInputSpec):
    atlas_configs = traits.Dict(
        desc=(
            "Atlas configs for atlases to run connectivity for. "
            "Keys are atlas names, values are dictionaries with at least the following keys: "
            "'labels' and 'dwi_resolution_mif'. "
            "'labels' is the parcellation labels file, and 'dwi_resolution_mif' is the "
            "parcellation file in the same resolution as the DWI data, in MIF format."
        ),
        mandatory=True,
    )
    tracking_params = traits.List(desc="list of sets of parameters for tck2connectome")


class MRTrixAtlasGraphOutputSpec(TraitedSpec):
    connectivity_matfile = File(exists=True)
    exemplar_files = OutputMultiObject(File(exists=True))
    commands = File()


class MRTrixAtlasGraph(SimpleInterface):
    """Produce one connectivity matrix per atlas based on MRtrix tractography"""

    input_spec = MRTrixAtlasGraphInputSpec
    output_spec = MRTrixAtlasGraphOutputSpec

    def _run_interface(self, runtime):
        # Get all inputs from the ApplyTransforms object
        cwd = runtime.cwd
        ifargs = self.inputs.get()
        nthreads = ifargs.get("nthreads", 1)
        atlas_configs = ifargs.pop("atlas_configs")
        tracking_params = self.inputs.tracking_params
        del ifargs["in_parc"]
        using_weights = isdefined(ifargs["in_weights"])
        c2t_args = {"input_tck_weights": ifargs["in_weights"]} if using_weights else {}
        c2t_args["nthreads"] = nthreads
        c2t_args["quiet"] = False
        ifargs["quiet"] = False
        ifargs["out_assignments"] = "assignments.txt"

        # Make a workflow for each atlas and tracking parameter set
        workflow = pe.Workflow(name="mrtrix_atlasgraph")
        nodes = []
        c2t_nodes = []
        num_nodes = len(tracking_params) * len(atlas_configs)
        merge_mats = pe.Node(niu.Merge(num_nodes), name="merge_mats")
        merge_csvs = pe.Node(niu.Merge(num_nodes), name="merge_csvs")
        merge_tcks = pe.Node(niu.Merge(num_nodes), name="merge_tcks")
        merge_weights = pe.Node(niu.Merge(num_nodes), name="merge_weights")
        merge_exemplars = pe.Node(niu.Merge(3), name="merge_exemplars")
        compress_exemplars = pe.Node(CompressConnectome2Tck(), name="compress_exemplars")
        outputnode = pe.Node(
            niu.IdentityInterface(fields=["matfiles", "tckfiles", "weights"]),
            name="outputnode",
        )
        workflow.connect([
            (merge_mats, outputnode, [("out", "matfiles")]),
            (merge_tcks, outputnode, [("out", "tckfiles")]),
            (merge_weights, outputnode, [("out", "weights")]),
        ])  # fmt:skip

        in_num = 1
        for atlas_name, atlas_config in atlas_configs.items():
            for tracking_param_set in self.inputs.tracking_params:
                node_args = deepcopy(ifargs)

                # These are overwritten in each node
                node_args.pop("tracking_params")

                measure_name = tracking_param_set["measure"]
                node_args.update(tracking_param_set)
                node_args["atlas_name"] = atlas_name
                node_args["atlas_labels_file"] = atlas_config["labels"]
                node_args["in_parc"] = atlas_config["dwi_resolution_mif"]
                nodes.append(
                    pe.Node(
                        BuildConnectome(**node_args),
                        name=f"{atlas_name}_{measure_name}",
                        n_procs=nthreads,
                    )
                )
                c2t_nodes.append(
                    pe.Node(
                        Connectome2Tck(
                            in_tck=node_args["in_file"],
                            in_parc=atlas_config["dwi_resolution_mif"],
                            output_files="single",
                            **c2t_args,
                        ),
                        name=f"{atlas_name}_{measure_name}_c2t",
                        n_procs=nthreads,
                    )
                )
                workflow.connect([
                    (nodes[-1], c2t_nodes[-1], [("out_assignments", "in_assignments")]),
                    (nodes[-1], merge_mats, [("connectivity_matfile", f"in{in_num}")]),
                    (nodes[-1], merge_csvs, [("out_file", f"in{in_num}")]),
                    (c2t_nodes[-1], merge_tcks, [("exemplar_tck", f"in{in_num}")]),
                ])  # fmt:skip
                if using_weights:
                    workflow.connect([
                        (c2t_nodes[-1], merge_weights, [("exemplar_weights", f"in{in_num}")]),
                    ])  # fmt:skip
                in_num += 1

        # Get the exemplar tcks and weights
        workflow.connect([
            (merge_tcks, merge_exemplars, [("out", "in1")]),
            (merge_weights, merge_exemplars, [("out", "in2")]),
            (merge_csvs, merge_exemplars, [("out", "in3")]),
            (merge_exemplars, compress_exemplars, [("out", "files")]),
            (compress_exemplars, outputnode, [("out_zip", "exemplar_files")])
        ])  # fmt:skip

        workflow.config["execution"]["stop_on_first_crash"] = "true"
        workflow.config["execution"]["remove_unnecessary_outputs"] = "false"
        workflow.base_dir = cwd
        if nthreads > 1:
            plugin_settings = {
                "plugin": "MultiProc",
                "plugin_args": {
                    "raise_insufficient": False,
                    "maxtasksperchild": 1,
                    "n_procs": nthreads,
                },
            }
            wf_result = workflow.run(**plugin_settings)
        else:
            wf_result = workflow.run()

        # Merge the connectivity matrices into a single file
        (merge_node,) = [
            node for node in list(wf_result.nodes) if node.name.endswith("merge_mats")
        ]
        merged_connectivity_file = op.join(cwd, "combined_connectivity.mat")
        _merge_conmats(merge_node.result.outputs.out, merged_connectivity_file)
        self._results["connectivity_matfile"] = merged_connectivity_file

        # Get the list of exemplars (+ weights if they exist)
        (compress_node,) = [
            node for node in list(wf_result.nodes) if node.name.endswith("compress_exemplars")
        ]
        rt = compress_node.run()
        self._results["exemplar_files"] = rt.outputs.out_zip

        return runtime


def _merge_conmats(matfile_list, outfile):
    """Merge the many matfiles output by dsi studio and ensure they conform"""
    connectivity_values = {}

    for matfile in matfile_list:
        connectivity_values.update(loadmat(matfile))
    savemat(outfile, connectivity_values, long_field_names=True, do_compression=True)


class _Connectome2TckInputSpec(MRTrix3BaseInputSpec):
    in_tck = File(exists=True, argstr="%s", position=0, mandatory=True, desc="input tck file")
    in_assignments = File(
        exists=True,
        argstr="%s",
        position=1,
        mandatory=True,
        desc="input tck assignments file",
    )
    out_prefix = traits.Str(
        name_source="in_parc",
        name_template="%s_exemplars.tck",
        argstr="%s",
        keep_extension=False,
        position=2,
        desc="this is a .tck file if '-files single'",
    )
    output_files = traits.Enum(
        "single",
        "per_edge",
        "per_node",
        default="single",
        argstr="-files %s",
    )
    input_tck_weights = File(exists=True, argstr="-tck_weights_in %s")
    output_tck_weights = File(
        name_source="input_tck_weights",
        name_template="%s",
        argstr="-prefix_tck_weights_out %s",
        keep_extension=False,
        requires=["input_tck_weights"],
    )
    in_parc = File(exists=True, argstr="-exemplars %s")
    keep_self = traits.Bool(argstr="-keep_self")
    quiet = traits.Bool(argstr="-quiet")


class _Connectome2TckOutputSpec(TraitedSpec):
    output_tck_weights = traits.Any()
    exemplar_tck = File(exists=True)
    out_prefix = traits.Any()
    exemplar_weights = File(exists=True)


class Connectome2Tck(MRTrix3Base):
    input_spec = _Connectome2TckInputSpec
    output_spec = _Connectome2TckOutputSpec
    _cmd = "connectome2tck"

    def _list_outputs(self):
        if not self.inputs.output_files == "single":
            raise NotImplementedError("Interface only supports single file output")
        outputs = super(Connectome2Tck, self)._list_outputs()
        exemplar_weights = outputs["output_tck_weights"] + ".csv"
        if op.exists(exemplar_weights):
            outputs["exemplar_weights"] = exemplar_weights
        exemplar_tck = outputs["out_prefix"]  # This is a tck file in single mode
        outputs["exemplar_tck"] = exemplar_tck
        return outputs


class _CompressConnectome2TckInputSpec(BaseInterfaceInputSpec):
    files = InputMultiObject(File(exists=True), mandatory=True)
    out_zip = File("connectome2tck.zip", usedefault=True, exists=False)


class _CompressConnectome2TckOutputSpec(TraitedSpec):
    out_zip = File(exists=True)


class CompressConnectome2Tck(SimpleInterface):
    input_spec = _CompressConnectome2TckInputSpec
    output_spec = _CompressConnectome2TckOutputSpec

    def _run_interface(self, runtime):
        out_zip = op.join(runtime.cwd, self.inputs.out_zip)
        zipfh = zipfile.ZipFile(out_zip, "w")
        # Get the matrix csvs and add them to the zip
        csvfiles = [
            fname
            for fname in self.inputs.files
            if fname.endswith(".csv") and not fname.endswith("weights.csv")
        ]
        for csvfile in csvfiles:
            zipfh.write(
                csvfile,
                arcname=_rename_connectome(csvfile, suffix="connectome.csv"),
                compresslevel=8,
                compress_type=zipfile.ZIP_DEFLATED,
            )

        # Get the sift weights if they exist
        weightfiles = [fname for fname in self.inputs.files if fname.endswith("weights.csv")]
        for weightfile in weightfiles:
            zipfh.write(
                weightfile,
                arcname=_rename_connectome(weightfile, suffix="_weights.csv"),
                compresslevel=8,
                compress_type=zipfile.ZIP_DEFLATED,
            )

        # Get the tck files
        tckfiles = [
            fname
            for fname in self.inputs.files
            if fname.endswith(".tck") or fname.endswith(".tck.gz")
        ]
        for tckfile in tckfiles:
            zipfh.write(
                tckfile,
                arcname=_rename_connectome(tckfile, suffix="_exemplars.tck"),
                compresslevel=8,
                compress_type=zipfile.ZIP_DEFLATED,
            )

        zipfh.close()
        self._results["out_zip"] = out_zip
        return runtime


def _rename_connectome(connectome_csv, suffix="_connectome.csv"):
    """
    >>> pth = "/a/b/c/qsirecon_wf/sub-X_mrtrix_multishell_msmt_fast/" \
    ...    "sub_X_ses_1_space_T1w_desc_preproc_recon_wf/mrtrix_conn/" \
    ...    "calc_connectivity/mrtrix_atlasgraph/" \
    ...    "schaefer200x17_sift_invnodevol_radius2_count/connectome.csv"
    >>> _rename_connectome(pth)
    "sub-X_ses-1_space-ACPC_desc-preproc_schaefer200x17_sift_invnodevol_radius2_count_connectome.csv"
    """
    parts = connectome_csv.split(os.sep)
    conn_name = parts[-2]
    try:
        (image_name,) = [
            part for part in parts if part.startswith("sub_") and part.endswith("recon_wf")
        ]
    except Exception:
        raise Exception(f"unable to detect image name from these parts {parts}")
    image_name = image_name[: -len("_recon_wf")]
    return "connectome2tck/" + _rebids(image_name) + "_" + conn_name + suffix


def _rebids(name):
    parts = name.split("_")
    bidsified = ""
    for partnum, part in enumerate(parts):
        bidsified += part
        bidsified += "-_"[partnum % 2]
    return bidsified[:-1]


class _ITKTransformConvertInputSpec(CommandLineInputSpec):
    in_transform = traits.File(exists=True, argstr="%s", mandatory=True, position=0)
    operation = traits.Enum(
        "itk_import", default="itk_import", usedefault=True, posision=1, argstr="%s"
    )
    out_transform = traits.File(
        argstr="%s",
        name_source="in_transform",
        name_template="%s.txt",
        keep_extension=False,
        position=-1,
    )


class _ITKTransformConvertOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ITKTransformConvert(CommandLine):
    _cmd = "transformconvert"
    input_spec = _ITKTransformConvertInputSpec
    output_spec = _ITKTransformConvertOutputSpec


class _TransformHeaderInputSpec(CommandLineInputSpec):
    transform_file = traits.File(exists=True, position=0, mandatory=True, argstr="-linear %s")
    in_image = traits.File(exists=True, mandatory=True, position=1, argstr="%s")
    out_image = traits.File(
        argstr="%s",
        name_source="in_image",
        name_template="%s_hdrxform.nii.gz",
        keep_extension=False,
        position=-1,
    )


class _TransformHeaderOutputSpec(TraitedSpec):
    out_image = File(exists=True)


class TransformHeader(CommandLine):
    input_spec = _TransformHeaderInputSpec
    output_spec = _TransformHeaderOutputSpec
    _cmd = "mrtransform -strides -1,-2,3"
