#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

import os.path as op
from glob import glob

import nibabel as nb
import numpy as np
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

LOGGER = logging.getLogger("nipype.interface")


class QSIPrepAnatomicalIngressInputSpec(BaseInterfaceInputSpec):
    recon_input_dir = traits.Directory(
        exists=True, mandatory=True, help="directory containing subject results directories"
    )
    subject_id = traits.Str()
    subjects_dir = File(exists=True)
    infant_mode = traits.Bool(mandatory=True)


class QSIPrepAnatomicalIngressOutputSpec(TraitedSpec):
    # sub-1_desc-aparcaseg_dseg.nii.gz
    t1_aparc = File()
    # sub-1_dseg.nii.gz
    t1_seg = File()
    # sub-1_desc-aseg_dseg.nii.gz
    t1_aseg = File()
    # sub-1_desc-brain_mask.nii.gz
    t1_brain_mask = File()
    # sub-1_desc-preproc_T1w.nii.gz
    t1_preproc = File()
    # sub-1_label-CSF_probseg.nii.gz
    t1_csf_probseg = File()
    # sub-1_label-GM_probseg.nii.gz
    t1_gm_probseg = File()
    # sub-1_label-WM_probseg.nii.gz
    t1_wm_probseg = File()
    # sub-1_from-orig_to-T1w_mode-image_xfm.txt
    orig_to_t1_mode_forward_transform = File()
    # sub-1_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
    t1_2_mni_reverse_transform = File()
    # sub-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
    t1_2_mni_forward_transform = File()


class QSIPrepAnatomicalIngress(SimpleInterface):
    """Get only the useful files from a QSIPrep anatomical output.

    Many of the preprocessed outputs aren't useful for reconstruction
    (mainly anything that has been mapped forward into template space).
    """

    input_spec = QSIPrepAnatomicalIngressInputSpec
    output_spec = QSIPrepAnatomicalIngressOutputSpec

    def _run_interface(self, runtime):
        # The path to the output from the qsirecon run
        sub = self.inputs.subject_id
        qp_root = op.join(self.inputs.recon_input_dir, "sub-" + sub)
        anat_root = op.join(qp_root, "anat")
        # space-T1w files
        self._get_if_exists(
            "t1_aparc",
            "%s/sub-%s*desc-aparcaseg_dseg.nii.*" % (anat_root, sub),
            excludes=["space-MNI"],
        )
        self._get_if_exists(
            "t1_seg", "%s/sub-%s_*dseg.nii*" % (anat_root, sub), excludes=["space-MNI", "aseg"]
        )
        self._get_if_exists(
            "t1_aseg",
            "%s/sub-%s_*aseg_dseg.nii*" % (anat_root, sub),
            excludes=["space-MNI", "aparc"],
        )
        self._get_if_exists(
            "t1_brain_mask",
            "%s/sub-%s*_desc-brain_mask.nii*" % (anat_root, sub),
            excludes=["space-MNI"],
        )
        self._get_if_exists(
            "t1_preproc",
            "%s/sub-%s_desc-preproc_T*w.nii*" % (anat_root, sub),
            excludes=["space-MNI"],
        )
        if "t1_preproc" not in self._results:
            LOGGER.warning("Unable to find a preprocessed T1w in %s", qp_root)
        self._get_if_exists(
            "t1_csf_probseg",
            "%s/sub-%s*_label-CSF_probseg.nii*" % (anat_root, sub),
            excludes=["space-MNI"],
        )
        self._get_if_exists(
            "t1_gm_probseg",
            "%s/sub-%s*_label-GM_probseg.nii*" % (anat_root, sub),
            excludes=["space-MNI"],
        )
        self._get_if_exists(
            "t1_wm_probseg",
            "%s/sub-%s*_label-WM_probseg.nii*" % (anat_root, sub),
            excludes=["space-MNI"],
        )
        self._get_if_exists(
            "orig_to_t1_mode_forward_transform",
            "%s/sub-%s*_from-orig_to-T*w_mode-image_xfm.txt" % (anat_root, sub),
        )
        self._get_if_exists(
            "t1_2_mni_reverse_transform",
            "%s/sub-%s*_from-MNI152NLin2009cAsym_to-T*w*_xfm.h5" % (anat_root, sub),
        )
        self._get_if_exists(
            "t1_2_mni_forward_transform",
            "%s/sub-%s*_from-T*w_to-MNI152NLin2009cAsym_mode-image_xfm.h5" % (anat_root, sub),
        )

        return runtime

    def _get_if_exists(self, name, pattern, excludes=None):
        files = glob(pattern)

        if excludes is not None:
            files = [
                fname
                for fname in files
                if not any([exclude in op.split(fname)[1] for exclude in excludes])
            ]

        if len(files) == 1:
            self._results[name] = files[0]


"""

The spherical harmonic coefficients are stored as follows. First, since the
signal attenuation profile is real, it has conjugate symmetry, i.e. Y(l,-m) =
Y(l,m)* (where * denotes the complex conjugate). Second, the diffusion profile
should be antipodally symmetric (i.e. S(x) = S(-x)), implying that all odd l
components should be zero. Therefore, only the even elements are computed. Note
that the spherical harmonics equations used here differ slightly from those
conventionally used, in that the (-1)^m factor has been omitted. This should be
taken into account in all subsequent calculations. Each volume in the output
image corresponds to a different spherical harmonic component.

Each volume will
correspond to the following:

volume 0: l = 0, m = 0 ;
volume 1: l = 2, m = -2 (imaginary part of m=2 SH) ;
volume 2: l = 2, m = -1 (imaginary part of m=1 SH)
volume 3: l = 2, m = 0 ;
volume 4: l = 2, m = 1 (real part of m=1 SH) ;
volume 5: l = 2, m = 2 (real part of m=2 SH) ; etc…


lmax = 2

vol	l	m
0	0	0
1	2	-2
2	2	-1
3	2	0
4	2	1
5	2	2

"""


lmax_lut = {6: 2, 15: 4, 28: 6, 45: 8}


def get_l_m(lmax):
    ell = []
    m = []
    for _ell in range(0, lmax + 1, 2):
        for _m in range(-_ell, _ell + 1):
            ell.append(_ell)
            m.append(_m)

    return np.array(ell), np.array(m)


def calculate_steinhardt(sh_l, sh_m, data, q_num):
    l_mask = sh_l == q_num
    images = data[..., l_mask]
    scalar = 4 * np.pi / (2 * q_num + 1)
    s_param = scalar * np.sum(images**2, 3)
    return np.sqrt(s_param)


class CalculateSOPInputSpec(BaseInterfaceInputSpec):
    sh_nifti = traits.File(mandatory=True, exists=True)
    order = traits.Enum(2, 4, 6, 8, default=6, usedefault=True)


class CalculateSOPOutputSpec(TraitedSpec):
    q2_file = traits.File()
    q4_file = traits.File()
    q6_file = traits.File()
    q8_file = traits.File()


class CalculateSOP(SimpleInterface):
    input_spec = CalculateSOPInputSpec
    output_spec = CalculateSOPOutputSpec

    def _run_interface(self, runtime):

        # load the input nifti image
        img = nb.load(self.inputs.sh_nifti)

        # determine what the lmax was based on the number of volumes
        num_vols = img.shape[3]
        if num_vols not in lmax_lut:
            raise ValueError("Not an SH image")
        lmax = lmax_lut[num_vols]

        # Do we have enough SH coeffs to calculate all the SOPs?
        if self.inputs.order > lmax:
            raise Exception(
                "Not enough SH coefficients (found {}) "
                "to calculate SOP order {}".format(num_vols, self.inputs.order)
            )
        sh_l, sh_m = get_l_m(lmax)
        sh_data = img.get_fdata()

        # Normalize the FODs so they integrate to 1
        sh_data = sh_data / sh_data[:, :, :, 0, None]

        # to get a specific order
        def calculate_order(order):
            out_fname = fname_presuffix(
                self.inputs.sh_nifti,
                suffix="q-%d_SOP.nii.gz" % order,
                use_ext=False,
                newpath=runtime.cwd,
            )
            order_data = calculate_steinhardt(sh_l, sh_m, sh_data, order)

            # Save with the new name in the sandbox
            nb.Nifti1Image(order_data, img.affine).to_filename(out_fname)
            self._results["q%d_file" % order] = out_fname

        # calculate!
        for order in range(2, self.inputs.order + 2, 2):
            calculate_order(order)

        return runtime


class _VoxelSizeChooserInputSpec(BaseInterfaceInputSpec):
    voxel_size = traits.Float()
    input_image = File(exists=True)
    anisotropic_strategy = traits.Enum("min", "max", "mean", usedefault=True)


class _VoxelSizeChooserOutputSpec(TraitedSpec):
    voxel_size = traits.Float()


class VoxelSizeChooser(SimpleInterface):
    input_spec = _VoxelSizeChooserInputSpec
    output_spec = _VoxelSizeChooserOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.input_image) and not isdefined(self.inputs.voxel_size):
            raise Exception("Either voxel_size or input_image need to be defined")

        # A voxel size was specified without an image
        if isdefined(self.inputs.voxel_size):
            voxel_size = self.inputs.voxel_size
        else:
            # An image was provided
            img = nb.load(self.inputs.input_image)
            zooms = img.header.get_zooms()[:3]
            if self.inputs.anisotropic_strategy == "min":
                voxel_size = min(zooms)
            elif self.inputs.anisotropic_strategy == "max":
                voxel_size = max(zooms)
            else:
                voxel_size = np.round(np.mean(zooms), 2)

        self._results["voxel_size"] = voxel_size
        return runtime


class _GetTemplateInputSpec(BaseInterfaceInputSpec):
    template_name = traits.Enum(
        "MNI152NLin2009cAsym",
        "MNIInfant",
        mandatory=True,
    )


class _GetTemplateOutputSpec(BaseInterfaceInputSpec):
    template_file = File(exists=True)
    mask_file = File(exists=True)


class GetTemplate(SimpleInterface):
    input_spec = _GetTemplateInputSpec
    output_spec = _GetTemplateOutputSpec

    def _run_interface(self, runtime):
        from templateflow.api import get as get_template

        template_file = str(
            get_template(
                self.inputs.template_name,
                cohort=[None, "2"],
                resolution="1",
                desc=None,
                suffix="T1w",
                extension=".nii.gz",
            ),
        )
        mask_file = str(
            get_template(
                self.inputs.template_name,
                cohort=[None, "2"],
                resolution="1",
                desc="brain",
                suffix="mask",
                extension=".nii.gz",
            ),
        )

        self._results["template_file"] = template_file
        self._results["mask_file"] = mask_file

        return runtime
