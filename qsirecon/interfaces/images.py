#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

import os
from subprocess import PIPE, Popen
from textwrap import indent

import nibabel as nb
import numpy as np
from dipy.io import read_bvals_bvecs
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
from niworkflows.interfaces.header import _ValidateImageInputSpec

from .mrtrix import SS3T_ROOT

LOGGER = logging.getLogger("nipype.interface")


class ConformDwiInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(mandatory=True, desc="dwi image")
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    orientation = traits.Enum("LPS", "LAS", default="LPS", usedefault=True)


class ConformDwiOutputSpec(TraitedSpec):
    dwi_file = File(exists=True, desc="conformed dwi image")
    bvec_file = File(exists=True, desc="conformed bvec file")
    bval_file = File(exists=True, desc="conformed bval file")
    out_report = File(exists=True, desc="HTML segment containing warning")


class ConformDwi(SimpleInterface):
    """Conform a series of dwi images to enable merging.
    Performs three basic functions:
    #. Orient image to requested orientation
    #. Validate the qform and sform, set qform code to 1
    #. Flip bvecs accordingly
    #. Do nothing to the bvals
    Note: This is not as nuanced as fmriprep's version
    """

    input_spec = ConformDwiInputSpec
    output_spec = ConformDwiOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.dwi_file
        orientation = self.inputs.orientation
        suffix = "_" + orientation
        out_fname = fname_presuffix(fname, suffix=suffix, newpath=runtime.cwd)

        # If not defined, find it
        if isdefined(self.inputs.bval_file):
            bval_fname = self.inputs.bval_file
        else:
            bval_fname = fname_presuffix(fname, suffix=".bval", use_ext=False)

        if isdefined(self.inputs.bvec_file):
            bvec_fname = self.inputs.bvec_file
        else:
            bvec_fname = fname_presuffix(fname, suffix=".bvec", use_ext=False)

        out_bvec_fname = fname_presuffix(bvec_fname, suffix=suffix, newpath=runtime.cwd)
        validator = ValidateImage(in_file=fname)
        validated = validator.run()
        self._results["out_report"] = validated.outputs.out_report
        input_img = nb.load(validated.outputs.out_file)

        input_axcodes = nb.aff2axcodes(input_img.affine)
        # Is the input image oriented how we want?
        new_axcodes = tuple(orientation)

        if not input_axcodes == new_axcodes:
            # Re-orient
            LOGGER.info("Re-orienting %s to %s", fname, orientation)
            input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
            desired_orientation = nb.orientations.axcodes2ornt(new_axcodes)
            transform_orientation = nb.orientations.ornt_transform(
                input_orientation, desired_orientation
            )
            reoriented_img = input_img.as_reoriented(transform_orientation)
            reoriented_img.to_filename(out_fname)
            self._results["dwi_file"] = out_fname

            # Flip the bvecs
            if os.path.exists(bvec_fname):
                LOGGER.info("Reorienting %s to %s", bvec_fname, orientation)
                bvec_array = np.loadtxt(bvec_fname)
                if not bvec_array.shape[0] == transform_orientation.shape[0]:
                    raise ValueError("Unrecognized bvec format")
                output_array = np.zeros_like(bvec_array)
                for this_axnum, (axnum, flip) in enumerate(transform_orientation):
                    output_array[this_axnum] = bvec_array[int(axnum)] * flip
                np.savetxt(out_bvec_fname, output_array, fmt="%.8f ")
                self._results["bvec_file"] = out_bvec_fname
                self._results["bval_file"] = bval_fname

        else:
            LOGGER.info("Not applying reorientation to %s: already in %s", fname, orientation)
            self._results["dwi_file"] = fname
            if os.path.exists(bvec_fname):
                self._results["bvec_file"] = bvec_fname
                self._results["bval_file"] = bval_fname

        return runtime


class ValidateImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="validated image")
    out_report = File(exists=True, desc="HTML segment containing warning")


class ValidateImage(SimpleInterface):
    """
    Check the correctness of x-form headers (matrix and code)
    This interface implements the `following logic
    <https://github.com/poldracklab/fmriprep/issues/873#issuecomment-349394544>`_:
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | valid quaternions | `qform_code > 0` | `sform_code > 0` | `qform == sform` \
| actions                                        |
    +===================+==================+==================+==================\
+================================================+
    | True              | True             | True             | True             \
| None                                           |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | True              | True             | False            | *                \
| sform, scode <- qform, qcode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | *                | True             | False            \
| qform, qcode <- sform, scode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | True             | *                \
| qform, qcode <- sform, scode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | False             | *                | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    """

    input_spec = _ValidateImageInputSpec
    output_spec = ValidateImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.join(runtime.cwd, "report.html")

        # Retrieve xform codes
        sform_code = int(img.header._structarr["sform_code"])
        qform_code = int(img.header._structarr["qform_code"])

        # Check qform is valid
        valid_qform = False
        try:
            qform = img.get_qform()
            valid_qform = True
        except ValueError:
            pass

        sform = img.get_sform()
        if np.linalg.det(sform) == 0:
            valid_sform = False
        else:
            RZS = sform[:3, :3]
            zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
            valid_sform = np.allclose(zooms, img.header.get_zooms()[:3])

        # Matching affines
        matching_affines = valid_qform and np.allclose(qform, sform)

        # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
        if matching_affines and qform_code > 0 and sform_code > 0:
            self._results["out_file"] = self.inputs.in_file
            open(out_report, "w").close()
            self._results["out_report"] = out_report
            return runtime

        # A new file will be written
        out_fname = fname_presuffix(self.inputs.in_file, suffix="_valid", newpath=runtime.cwd)
        self._results["out_file"] = out_fname

        # Row 2:
        if valid_qform and qform_code > 0 and (sform_code == 0 or not valid_sform):
            img.set_sform(qform, qform_code)
            warning_txt = "Note on orientation: sform matrix set"
            description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
        # Rows 3-4:
        # Note: if qform is not valid, matching_affines is False
        elif (valid_sform and sform_code > 0) and (not matching_affines or qform_code == 0):
            img.set_qform(img.get_sform(), sform_code)
            warning_txt = "Note on orientation: qform matrix overwritten"
            description = """\
<p class="elem-desc">The qform has been copied from sform.</p>
"""
            if not valid_qform and qform_code > 0:
                warning_txt = "WARNING - Invalid qform information"
                description = """\
<p class="elem-desc">
    The qform matrix found in the file header is invalid.
    The qform has been copied from sform.
    Checking the original qform information from the data produced
    by the scanner is advised.
</p>
"""
        # Rows 5-6:
        else:
            affine = img.header.get_base_affine()
            img.set_sform(affine, nb.nifti1.xform_codes["scanner"])
            img.set_qform(affine, nb.nifti1.xform_codes["scanner"])
            warning_txt = "WARNING - Missing orientation information"
            description = """\
<p class="elem-desc">
    QSIRecon could not retrieve orientation information from the image header.
    The qform and sform matrices have been set to a default, LAS-oriented affine.
    Analyses of this dataset MAY BE INVALID.
</p>
"""
        snippet = '<h3 class="elem-title">%s</h3>\n%s:\n\t %s\n' % (
            warning_txt,
            self.inputs.in_file,
            description,
        )
        # Store new file and report
        img.to_filename(out_fname)
        with open(out_report, "w") as fobj:
            fobj.write(indent(snippet, "\t" * 3))

        self._results["out_report"] = out_report
        return runtime


def bvec_to_rasb(bval_file, bvec_file, img_file, workdir):
    """Use mrinfo to convert a bvec to RAS+ world coordinate reference frame"""

    # Make a temporary bvec file that mrtrix likes
    temp_bvec = fname_presuffix(bvec_file, suffix="_fix", newpath=workdir)
    lps_bvec = np.loadtxt(bvec_file).reshape(3, -1)
    np.savetxt(temp_bvec, lps_bvec * np.array([[-1], [1], [1]]))

    # Run mrinfo to to get the RAS+ vector
    cmd = [SS3T_ROOT + "/mrinfo", "-dwgrad", "-fslgrad", temp_bvec, bval_file, img_file]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    LOGGER.info(" ".join(cmd))
    if err:
        raise Exception(str(err))

    return np.fromstring(out, dtype=float, sep=" ")[:3]


def split_bvals_bvecs(bval_file, bvec_file, img_files, deoblique, working_dir):
    """Split bvals and bvecs into one text file per image."""
    if deoblique:
        LOGGER.info("Converting oblique-image bvecs to world coordinate reference frame")
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    split_bval_files = []
    split_bvec_files = []
    for nsample, (bval, bvec, img_file) in enumerate(zip(bvals[:, None], bvecs, img_files)):
        bval_fname = fname_presuffix(bval_file, suffix="_%04d" % nsample, newpath=working_dir)
        bvec_suffix = "_ortho_%04d" % nsample if not deoblique else "_%04d" % nsample
        bvec_fname = fname_presuffix(bvec_file, bvec_suffix, newpath=working_dir)
        np.savetxt(bval_fname, bval)
        np.savetxt(bvec_fname, bvec)

        # re-write the bvec deobliqued, if requested
        if deoblique:
            rasb = bvec_to_rasb(bval_fname, bvec_fname, img_file, working_dir)
            # Convert to image axis orientation
            ornt = nb.aff2axcodes(nb.load(img_file).affine)
            flippage = np.array([1 if ornt[n] == "RAS"[n] else -1 for n in [0, 1, 2]])
            deobliqued_bvec = rasb * flippage
            np.savetxt(bvec_fname, deobliqued_bvec)

        split_bval_files.append(bval_fname)
        split_bvec_files.append(bvec_fname)

    return split_bval_files, split_bvec_files


def to_lps(input_img, new_axcodes=("L", "P", "S")):
    if isinstance(input_img, str):
        input_img = nb.load(input_img)
    input_axcodes = nb.aff2axcodes(input_img.affine)
    # Is the input image oriented how we want?
    if not input_axcodes == new_axcodes:
        # Re-orient
        input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
        desired_orientation = nb.orientations.axcodes2ornt(new_axcodes)
        transform_orientation = nb.orientations.ornt_transform(
            input_orientation, desired_orientation
        )
        reoriented_img = input_img.as_reoriented(transform_orientation)
        return reoriented_img
    else:
        return input_img
