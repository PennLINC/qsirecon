#!python
"""
Wrappers for the TORTOISE programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import logging
import os
import os.path as op
import subprocess

import nilearn.image as nim
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from ..utils.boilerplate import ConditionalDoc

LOGGER = logging.getLogger("nipype.interface")

SLOPPY_DRBUDDI = (
    "--DRBUDDI_stage "
    "\[learning_rate=\{0.3\},cfs=\{100:8:4\},field_smoothing=\{9:0\},"
    "metrics=\{MSJac:CC\},restrict_constrain=\{1:1\}\] "
)


class TORTOISEInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(desc="number of OMP threads")


class TORTOISECommandLine(CommandLine):
    """Support for TORTOISE commands that utilize OpenMP
    Sets the environment variable 'OMP_NUM_THREADS' to the number
    of threads specified by the input num_threads.
    """

    input_spec = TORTOISEInputSpec
    _num_threads = None

    def __init__(self, **inputs):
        super(TORTOISECommandLine, self).__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, "num_threads")
        if not self._num_threads:
            self._num_threads = os.environ.get("OMP_NUM_THREADS", None)
            if not self._num_threads:
                self._num_threads = os.environ.get("NSLOTS", None)
        if not isdefined(self.inputs.num_threads) and self._num_threads:
            self.inputs.num_threads = int(self._num_threads)
        self._num_threads_update()

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({"OMP_NUM_THREADS": str(self.inputs.num_threads)})

    def run(self, **inputs):
        if "num_threads" in inputs:
            self.inputs.num_threads = inputs["num_threads"]
        self._num_threads_update()
        return super(TORTOISECommandLine, self).run(**inputs)


class _DRBUDDIInputSpec(TORTOISEInputSpec):
    num_threads = traits.Int(
        desc="number of OMP threads",
        argstr="--ncores %d",
        help="Number of cores to use in the CPU version. The default is 50% of system cores.",
        nohash=True,
    )
    blip_up_image = File(
        exists=True,
        help="Full path to the input UP NIFTI file to be corrected.",
        argstr="-u %s",
        mandatory=True,
        copyfile=True,
    )
    blip_up_bmat = File(
        exists=True,
        help="Full path to the input UP NIFTI bmtxt file.",
        mandatory=False,
        copyfile=True,
    )
    blip_up_json = File(
        exists=True,
        help="Phase encoding information will be read from this",
        argstr="--up_json %s",
        mandatory=True,
        copyfile=True,
    )
    blip_down_image = File(
        exists=True,
        help="Full path to the input DOWN NIFTI file to be corrected.",
        argstr="-d %s",
        mandatory=True,
        copyfile=True,
    )
    blip_down_bmat = File(
        exists=True,
        help="Full path to the input DOWN NIFTI bmtxt file.",
        mandatory=False,
        copyfile=True,
    )
    structural_image = InputMultiObject(
        File(exists=True, copyfile=False),
        argstr="-s %s",
        help="Path(s) to anatomical image files. Can provide more than one. NO T1W's!!",
    )
    fieldmap_type = traits.Enum("epi", "rpe_series", mandatory=True)
    blip_assignments = traits.List()
    tensor_fit_bval_max = traits.Int(
        0,
        argstr="--DRBUDDI_DWI_bval_tensor_fitting %d",
        desc="Up to which b-value should be used for DRBUDDI's tensor fitting. "
        "Default: 0 , meaning use all b-values",
    )
    disable_initial_rigid = traits.Bool(
        False,
        argstr="--DRBUDDI_disable_initial_rigid %d",
        desc="DRBUDDI performs an initial registration between the up and down data."
        "This registration starts with rigid, followed by a quick diffeomorphic "
        "and finalized by another rigid. This parameter, when set to 1 disables "
        "all these registrations. Default: False",
    )
    start_with_diffeomorphic_for_rigid_reg = traits.Bool(
        False,
        argstr="--DRBUDDI_start_with_diffeomorphic_for_rigid_reg",
        desc="DRBUDDI performs an initial registration between the up and down data. "
        "This registration starts with rigid, followed by a quick diffeomorphic "
        "and finalized by another rigid. This parameter, when set to 1 disables "
        "the very initial rigid registration and starts with the quick diffemorphic. "
        "This is helpful with VERY DISTORTED data, for which the initial rigid "
        "registration is problematic. Default: False",
    )
    estimate_learning_rate_per_iteration = traits.Bool(
        False,
        argstr="--DRBUDDI_estimate_LR_per_iteration %d",
        desc="Flat to estimate learning rate at every iteration. "
        "Makes DRBUDDI slower but better results. Default: False",
    )
    sloppy = traits.Bool(
        False, argstr=SLOPPY_DRBUDDI, desc="use underpowered (sloppy) registration for speed"
    )
    disable_itk_threads = traits.Bool(True, usedefault=True, argstr="--disable_itk_threads")


class _DRBUDDIOutputSpec(TraitedSpec):
    # Direct outputs from DRBUDDI
    undistorted_reference = File(exists=True)
    bdown_to_bup_rigid_trans_h5 = File(exists=True)
    undistorted_reference = File(exists=True)
    blip_down_b0 = File(exists=True)
    blip_down_b0_corrected = File(exists=True)
    blip_down_b0_corrected_jac = File(exists=True)
    blip_down_b0_quad = File(exists=True)
    blip_up_b0 = File(exists=True)
    blip_up_b0_corrected = File(exists=True)
    blip_up_b0_corrected_jac = File(exists=True)
    blip_up_b0_quad = File(exists=True)
    deformation_finv = File(exists=True)
    deformation_minv = File(exists=True)
    blip_up_FA = File(exists=True)
    blip_down_FA = File(exists=True)
    structural_image = File(exists=True)


class DRBUDDI(TORTOISECommandLine):
    input_spec = _DRBUDDIInputSpec
    output_spec = _DRBUDDIOutputSpec
    _cmd = "DRBUDDI"

    def _format_arg(self, name, spec, value):
        """Trick to get blip_down_bmat symlinked without an arg"""
        if name in ("blip_down_bmat", "blip_up_bmat"):
            return ""
        if name == "structural_image":
            return "-s " + " ".join(value)
        return super(DRBUDDI, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["undistorted_reference"] = op.abspath("b0_corrected_final.nii")
        outputs["blip_down_b0"] = op.abspath("blip_down_b0.nii")
        outputs["blip_down_b0_corrected"] = op.abspath("blip_down_b0_corrected.nii")
        outputs["blip_down_b0_corrected_jac"] = op.abspath("blip_down_b0_corrected_JAC.nii")
        outputs["blip_down_b0_quad"] = op.abspath("blip_down_b0_quad.nii")
        outputs["blip_up_b0"] = op.abspath("blip_up_b0.nii")
        outputs["blip_up_b0_corrected"] = op.abspath("blip_up_b0_corrected.nii")
        outputs["blip_up_b0_corrected_jac"] = op.abspath("blip_up_b0_corrected_JAC.nii")
        outputs["blip_up_b0_quad"] = op.abspath("blip_up_b0_quad.nii")
        outputs["deformation_finv"] = op.abspath("deformation_FINV.nii.gz")
        outputs["deformation_minv"] = op.abspath("deformation_MINV.nii.gz")

        # There will be an hdf5 transform file if there is an initial rigid
        if not self.inputs.disable_initial_rigid:
            outputs["bdown_to_bup_rigid_trans_h5"] = op.abspath("bdown_to_bup_rigidtrans.hdf5")

        # There will be FA images created if two DWI series were used as inputs
        if self.inputs.fieldmap_type == "rpe_series":
            outputs["blip_up_FA"] = op.abspath("blip_up_FA.nii")
            outputs["blip_down_FA"] = op.abspath("blip_down_FA.nii")

        # If there was a T2w
        if self.inputs.structural_image:
            outputs["structural_image"] = op.abspath("structural_used.nii")
        return outputs


class _TORTOISEConvertInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True, copyfile=True)
    bvec_file = File(exists=True, mandatory=True, copyfile=True)
    dwi_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)


class _TORTOISEConvertOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    mask_file = File(exists=True)
    bmtxt_file = File(exists=True)


class TORTOISEConvert(SimpleInterface):
    input_spec = _TORTOISEConvertInputSpec
    output_spec = _TORTOISEConvertOutputSpec

    def _run_interface(self, runtime):
        """Convert gzipped niftis and bval/bvec into TORTOISE format."""
        dwi_file = fname_presuffix(
            self.inputs.dwi_file, newpath=runtime.cwd, use_ext=False, suffix=".nii"
        )
        dwi_img = nim.load_img(self.inputs.dwi_file, dtype="float32")
        dwi_img.set_data_dtype("float32")
        dwi_img.to_filename(dwi_file)
        bmtxt_file = make_bmat_file(self.inputs.bval_file, self.inputs.bvec_file)

        if isdefined(self.inputs.mask_file):
            mask_file = fname_presuffix(
                self.inputs.mask_file, newpath=runtime.cwd, use_ext=False, suffix=".nii"
            )
            mask_img = nim.load_img(self.inputs.mask_file, dtype="float32")
            mask_img.set_data_dtype("float32")
            mask_img.to_filename(mask_file)
            self._results["mask_file"] = mask_file

        self._results["dwi_file"] = dwi_file
        self._results["bmtxt_file"] = bmtxt_file

        return runtime


class TORTOISEReconCommandLine(TORTOISECommandLine):
    _link_me = ["bmtxt_file"]

    # Most TORTOISE commandline programs don't offer an option for
    # what the output file should be called. The input file
    def _list_outputs(self):

        # If this is a special case where in_file is not defined and
        # there is no _suffix_map, do the normal version
        if not hasattr(self.inputs, "in_file"):
            raise Exception("TORTOISEReconCommandLine requires an in_file")

        outputs = self.output_spec().get()
        if not self._suffix_map:
            raise Exception("Compute classes need to have a _suffix_map")

        for trait_name, suffix in self._suffix_map.items():
            new_fname = op.basename(self.inputs.in_file.replace(".nii", suffix + ".nii"))
            if op.exists(new_fname):
                outputs[trait_name] = op.abspath(new_fname)
        return outputs

    def _format_arg(self, name, spec, value):
        if not hasattr(self, "_link_me"):
            raise Exception
        if name in self._link_me:
            return ""
        return super(TORTOISEReconCommandLine, self)._format_arg(name, spec, value)


class _TORTOISEEstimatorInputSpec(TORTOISEInputSpec):
    in_file = File(
        exists=False,
        mandatory=True,
        argstr="--input %s",
        desc="Full path to the input NIFTI DWI",
        copyfile=False,
    )
    bmtxt_file = File(
        exists=True, mandatory=True, desc="Full path to the input bmtxt file", copyfile=False
    )
    # This is an example of a trait where we won't know if it's used until runtime
    mask = File(exists=True, argstr="--mask %s", desc="Full path to the mask NIFTI image")
    bval_cutoff = traits.CInt(
        argstr="--bval_cutoff %d",
        desc="Maximum b-value volumes to use for tensor fitting. (Default: use all volumes)",
        doc=ConditionalDoc(
            "A maximum b-value cutoff of b={value} was used.",
            if_false="All b-values were used for tensor fitting."),
    )
    inclusion_file = File(exists=True, argstr="--inclusion %s")
    voxelwise_bmat_file = File(
        exists=True, desc="Use voxelwise Bmatrices for gradient non-linearity correction"
    )


class _EstimateTensorInputSpec(_TORTOISEEstimatorInputSpec):
    reg_mode = traits.Enum(
        "WLLS",
        "NLLS",
        "RESTORE",
        "DIAG",
        "N2",
        "NT2",
        argstr="--reg_mode %s",
        usedefault=True,
        desc="Regression mode. WLLS: Weighted linear least squares, "
        "NLLS: Nonlinear least squares, "
        "RESTORE: Robust NLLS, "
        "DIAG: Diagonal Only NLLS, "
        "N2: Full diffusion tensor + free water NLLS, "
        "NT2: One full parenchymal diffusion tensor + one full flow tensor",
        doc=ConditionalDoc("Tensor fitting was performed with {value} regularization."),
    )
    free_water_diffusivity = traits.CInt(
        default_value=3000,
        argstr="--free_water_diffusivity %d",
        desc="Free water diffusivity in (mu m)^2/s for N2 fitting.",
        doc=ConditionalDoc(
            "Free water diffusivity was set to {value} (mu m)^2/s.",
            if_false="Free water diffusivity was set to the TORTOISE default of 3000 (mu m)^2/s."),
    )
    write_cs = traits.Bool(
        default_value=True,
        usedefault=True,
        argstr="--write_CS %d",
        desc="Write the Chi-squred image?",
        doc=ConditionalDoc("The Chi-squared image was written."),
    )
    noise_file = File(
        exists=True,
        copyfile=False,
        desc="Use this image for weigthing and correction of interpolation artifacts.",
    )


class _EstimateTensorOutputSpec(TraitedSpec):
    dt_file = File(exists=True)
    am_file = File(exists=True)
    cs_file = File(exists=True)


class EstimateTensor(TORTOISEReconCommandLine):
    input_spec = _EstimateTensorInputSpec
    output_spec = _EstimateTensorOutputSpec
    _cmd = "EstimateTensor"
    _suffix_map = {"dt_file": "_L1_DT", "am_file": "_L1_AM"}
    _link_me = [
        "bmtxt",
    ]

    def _format_arg(self, name, spec, value):
        """Trick to get noise image or voxelwise bmat symlinked without an arg"""
        if name == "noise_file":
            return "--use_noise 1"
        if name == "voxelwise_bmat_file":
            return "--use_voxelwise_bmat 1"
        return super(EstimateTensor, self)._format_arg(name, spec, value)


class _TensorMapInputSpec(TORTOISEInputSpec):
    in_file = File(exists=True, mandatory=True, argstr="%s", position=1, copyfile=True)
    am_file = File(exists=True, copyfile=False)


class _TensorMapCmdline(TORTOISEReconCommandLine):
    _link_me = ["am_file"]


class _ComputeFAMapInputSpec(_TensorMapInputSpec):
    filter_outliers = traits.Bool(
        True,
        usedefault=True,
        argstr="%d",
        position=2,
        doc=ConditionalDoc(
            "When calculating FA, negative eigenvalues were replaced with "
            "best estimate positive values. Affected voxels had their "
            "FA set to the median FA value of spatial neighbors. This imputation "
            "was not performed when calculating other tensor maps."),
    )


class _ComputeFAMapOutputSpec(TraitedSpec):
    fa_file = File(exists=True)


class ComputeFAMap(_TensorMapCmdline):
    input_spec = _ComputeFAMapInputSpec
    output_spec = _ComputeFAMapOutputSpec
    _cmd = "ComputeFAMap"
    _suffix_map = {"fa_file": "_FA"}


class _ComputeRDMapOutputSpec(TraitedSpec):
    rd_file = File(exists=True)


class ComputeRDMap(_TensorMapCmdline):
    input_spec = _TensorMapInputSpec
    output_spec = _ComputeRDMapOutputSpec
    _cmd = "ComputeRDMap"
    _suffix_map = {"rd_file": "_RD"}


class _ComputeADMapOutputSpec(TraitedSpec):
    ad_file = File(exists=True)


class ComputeADMap(_TensorMapCmdline):
    input_spec = _TensorMapInputSpec
    output_spec = _ComputeADMapOutputSpec
    _cmd = "ComputeADMap"
    _suffix_map = {"ad_file": "_AD"}


class _ComputeLIMapOutputSpec(TraitedSpec):
    li_file = File(exists=True)


class ComputeLIMap(_TensorMapCmdline):
    input_spec = _TensorMapInputSpec
    output_spec = _ComputeLIMapOutputSpec
    _cmd = "ComputeLIMap"
    _suffix_map = {"li_file": "_LI"}


class _EstimateMAPMRIInputSpec(_TORTOISEEstimatorInputSpec):
    dt_file = File(
        exists=True, argstr="--dti %s", requires=["a0_file"], desc="DTI image computed externally"
    )
    a0_file = File(exists=True, argstr="--A0 %s", desc="A0 image computed externally")
    map_order = traits.Int(
        default_value=4,
        usedefault=True,
        argstr="--map_order %d",
        desc="MAPMRI order",
        doc=ConditionalDoc("MAPMRI order was set to {value}."),
    )
    big_delta = traits.CFloat(argstr="--big_delta %.7f", desc="Big Delta in seconds")
    small_delta = traits.CFloat(argstr="--small_delta %.7f", desc="Small Delta in seconds")


class _EstimateMAPMRIOutputSpec(TraitedSpec):
    coeffs_file = File(exists=True)
    uvec_file = File(exists=True)


class EstimateMAPMRI(TORTOISEReconCommandLine):
    input_spec = _EstimateMAPMRIInputSpec
    output_spec = _EstimateMAPMRIOutputSpec
    _cmd = "EstimateMAPMRI"
    _suffix_map = {"coeffs_file": "_mapmri", "uvec_file": "_uvec"}


class _ComputeMAPMRIInputSpec(TORTOISEInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="_mapmri.nii file",
        copyfile=False,
    )
    uvec_file = File(exists=True, mandatory=True, argstr="%s", position=2, copyfile=False)


class _ComputeMAPMRI_PAOutputSpec(TraitedSpec):
    pa_file = File(exists=True)
    path_file = File(exists=True)


class ComputeMAPMRI_PA(TORTOISEReconCommandLine):
    input_spec = _ComputeMAPMRIInputSpec
    output_spec = _ComputeMAPMRI_PAOutputSpec
    _cmd = "ComputeMAPMRI_PA"
    _suffix_map = {"pa_file": "_PA", "path_file": "_PAth"}


class _ComputeMAPMRI_RTOPOutputSpec(TraitedSpec):
    rtop_file = File(exists=True)
    rtap_file = File(exists=True)
    rtpp_file = File(exists=True)


class ComputeMAPMRI_RTOP(TORTOISEReconCommandLine):
    input_spec = _ComputeMAPMRIInputSpec
    output_spec = _ComputeMAPMRI_RTOPOutputSpec
    _cmd = "ComputeMAPMRI_RTOP"
    _suffix_map = {"rtap_file": "_RTAP", "rtop_file": "_RTOP", "rtpp_file": "_RTPP"}


class _ComputeMAPMRI_NGOutputSpec(TraitedSpec):
    ng_file = File(exists=True)
    ngpar_file = File(exists=True)
    ngperp_file = File(exists=True)


class ComputeMAPMRI_NG(TORTOISEReconCommandLine):
    input_spec = _ComputeMAPMRIInputSpec
    output_spec = _ComputeMAPMRI_NGOutputSpec
    _cmd = "ComputeMAPMRI_NG"
    _suffix_map = {"ng_file": "_NG", "ngpar_file": "_NGpar", "ngperp_file": "_NGperp"}


def make_bmat_file(bvals, bvecs):
    pout = subprocess.run(["FSLBVecsToTORTOISEBmatrix", op.abspath(bvals), op.abspath(bvecs)])
    print(pout)
    return bvals.replace("bval", "bmtxt")
