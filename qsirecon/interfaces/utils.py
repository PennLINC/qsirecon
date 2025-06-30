#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utilities
^^^^^^^^^^^^^^^^^^^^^^^

"""

import json
import os

import nibabel as nb
from nilearn import image, plotting
from nipype import logging
from nipype.interfaces import ants
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.interfaces.header import ValidateImage

IFLOGGER = logging.getLogger("nipype.interfaces")


class _WarpConnectivityAtlasesInputSpec(BaseInterfaceInputSpec):
    atlas_configs = traits.Dict(
        mandatory=True,
        desc=(
            "Dictionary of atlas configurations. "
            "Keys are atlas names and values are dictionaries with the following keys: "
            "'file', 'label', 'metadata'. "
            "'file' is the path to the atlas file. "
            "'label' is the path to the label file. "
            "'metadata' is a dictionary with relevant metadata. "
            "'xfm_to_anat' is the path to the transform to get the atlas into T1w space."
        ),
    )
    reference_image = File(exists=True, desc="")
    space = traits.Str("T1w", usedefault=True)


class _WarpConnectivityAtlasesOutputSpec(TraitedSpec):
    atlas_configs = traits.Dict(
        desc=(
            "Dictionary of atlas configurations. "
            "This interface adds the following keys: "
            "'dwi_resolution_file', 'dwi_resolution_mif', 'orig_lut', 'mrtrix_lut'. "
            "The values are the paths to the transformed atlas files and the label files."
        ),
    )
    commands = File()


class WarpConnectivityAtlases(SimpleInterface):
    input_spec = _WarpConnectivityAtlasesInputSpec
    output_spec = _WarpConnectivityAtlasesOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.space == "T1w":
            transforms = [cfg["xfm_to_anat"] for cfg in self.inputs.atlas_configs.values()]
            if not all(os.path.isfile(transform) for transform in transforms):
                raise ValueError("No standard to T1w transform found in anatomical directory.")

        else:
            transforms = ["identity"] * len(self.inputs.atlas_configs)

        # Transform atlases to match the DWI data
        atlas_configs = self.inputs.atlas_configs.copy()
        resample_commands = []
        for i_atlas, (atlas_name, atlas_config) in enumerate(atlas_configs.items()):
            output_name = fname_presuffix(
                atlas_config["image"],
                newpath=runtime.cwd,
                suffix="_to_dwi",
            )
            output_mif = fname_presuffix(
                atlas_config["image"],
                newpath=runtime.cwd,
                suffix="_to_dwi.mif",
                use_ext=False,
            )
            output_mif_txt = fname_presuffix(
                atlas_config["image"],
                newpath=runtime.cwd,
                suffix="_mrtrixlabels.txt",
                use_ext=False,
            )
            output_orig_txt = fname_presuffix(
                atlas_config["image"],
                newpath=runtime.cwd,
                suffix="_origlabels.txt",
                use_ext=False,
            )

            atlas_configs[atlas_name]["dwi_resolution_file"] = output_name
            atlas_configs[atlas_name]["dwi_resolution_mif"] = output_mif
            atlas_configs[atlas_name]["orig_lut"] = output_mif_txt
            atlas_configs[atlas_name]["mrtrix_lut"] = output_orig_txt

            conform_atlas = ConformAtlas(in_file=atlas_config["image"], orientation="LPS")
            result = conform_atlas.run()
            lps_name = result.outputs.out_file

            resample_commands.append(
                _resample_atlas(
                    input_atlas=lps_name,
                    output_atlas=output_name,
                    transform=transforms[i_atlas],
                    ref_image=self.inputs.reference_image,
                )
            )
            label_convert(
                output_name,
                output_mif,
                output_orig_txt,
                output_mif_txt,
                atlas_config["labels"],
            )

        self._results["atlas_configs"] = atlas_configs
        commands_file = os.path.join(runtime.cwd, "transform_commands.txt")
        with open(commands_file, "w") as f:
            f.write("\n".join(resample_commands))

        self._results["commands"] = commands_file
        return runtime


def _resample_atlas(input_atlas, output_atlas, transform, ref_image):
    xform = ants.ApplyTransforms(
        transforms=[transform],
        reference_image=ref_image,
        input_image=input_atlas,
        output_image=output_atlas,
        interpolation="MultiLabel",
    )
    result = xform.run()

    return result.runtime.cmdline


def label_convert(original_atlas, output_mif, orig_txt, mrtrix_txt, atlas_labels_file):
    """Create a mrtrix label file from an atlas."""
    import pandas as pd

    atlas_labels_df = pd.read_table(atlas_labels_file)
    index_label_pairs = zip(atlas_labels_df["index"], atlas_labels_df["label"])
    orig_str = ""
    mrtrix_str = ""
    for i_row, (index, label) in enumerate(index_label_pairs):
        orig_str += f"{index}\t{label}\n"
        mrtrix_str += f"{i_row + 1}\t{label}\n"

    with open(mrtrix_txt, "w") as mrtrix_f:
        mrtrix_f.write(mrtrix_str)

    with open(orig_txt, "w") as orig_f:
        orig_f.write(orig_str)

    cmd = ["labelconvert", original_atlas, orig_txt, mrtrix_txt, output_mif]
    os.system(" ".join(cmd))
    if not os.path.isfile(output_mif):
        raise RuntimeError(f"Failed to create mrtrix label file from {original_atlas}")


class _ConformAtlasInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc="dwi image")
    orientation = traits.Enum("LPS", "LAS", default="LPS", usedefault=True)


class _ConformAtlasOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="conformed dwi image")


class ConformAtlas(SimpleInterface):
    """Conform a series of dwi images to enable merging.

    Performs three basic functions:
    #. Orient image to requested orientation
    #. Validate the qform and sform, set qform code to 1
    """

    input_spec = _ConformAtlasInputSpec
    output_spec = _ConformAtlasOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.in_file
        orientation = self.inputs.orientation
        suffix = "_" + orientation
        out_fname = fname_presuffix(fname, suffix=suffix, newpath=runtime.cwd)

        validator = ValidateImage(in_file=fname)
        validated = validator.run()
        input_img = nb.load(validated.outputs.out_file)

        input_axcodes = nb.aff2axcodes(input_img.affine)
        # Is the input image oriented how we want?
        new_axcodes = tuple(orientation)

        if not input_axcodes == new_axcodes:
            # Re-orient
            input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
            desired_orientation = nb.orientations.axcodes2ornt(new_axcodes)
            transform_orientation = nb.orientations.ornt_transform(
                input_orientation, desired_orientation
            )
            reoriented_img = input_img.as_reoriented(transform_orientation)
            reoriented_img.to_filename(out_fname)
            self._results["out_file"] = out_fname

        else:
            self._results["out_file"] = fname

        return runtime


class _WriteSidecarInputSpec(BaseInterfaceInputSpec):
    metadata = traits.Dict()


class _WriteSidecarOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class WriteSidecar(SimpleInterface):
    input_spec = _WriteSidecarInputSpec
    output_spec = _WriteSidecarOutputSpec

    def _run_interface(self, runtime):
        out_file = os.path.join(runtime.cwd, "sidecar.json")
        with open(out_file, "w") as outf:
            json.dump(self.inputs.metadata, outf)
        self._results["out_file"] = out_file
        return runtime


class _TestReportPlotInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)


class _TestReportPlotOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class TestReportPlot(SimpleInterface):
    input_spec = _TestReportPlotInputSpec
    output_spec = _TestReportPlotOutputSpec

    def _run_interface(self, runtime):
        img = image.index_img(self.inputs.dwi_file, 0)
        out_file = os.path.join(runtime.cwd, "brainfig.png")
        plotting.plot_img(
            img=img, output_file=out_file, title=os.path.basename(self.inputs.dwi_file)
        )
        self._results["out_file"] = out_file
        return runtime
