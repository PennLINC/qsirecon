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
            transforms = [
                cfg["xfm_to_anat"] for cfg in self.inputs.atlas_configs.values()
            ]
            if not all(os.path.isfile(transform) for transform in transforms):
                raise ValueError(
                    "No standard to T1w transform found in anatomical directory."
                )

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

            conform_atlas = ConformAtlas(
                in_file=atlas_config["image"], orientation="LPS"
            )
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
    """Create a mrtrix label file from an atlas.

    Parameters
    ----------
    original_atlas : str
        Path to the original atlas file, in NIfTI format.
    output_mif : str
        Path to the output mrtrix label file (mif[.gz]) to be written out.
    orig_txt : str
        Path to the output original label file (txt) to be written out.
    mrtrix_txt : str
        Path to the output mrtrix label file (txt) to be written out.
    atlas_labels_file : str
        Path to the atlas labels file (txt) to be read in.
        This file should have at least two columns: index and label.
    """
    import subprocess

    import pandas as pd

    atlas_labels_df = pd.read_table(atlas_labels_file)

    atlas_labels_df["index"] = atlas_labels_df["index"].astype(int)
    if 0 in atlas_labels_df["index"].values:
        print(
            f"WARNING: Atlas {atlas_labels_file} has a 0 index. This index will be dropped."
        )
        atlas_labels_df = atlas_labels_df.loc[atlas_labels_df["index"] != 0]

    index_label_pairs = zip(atlas_labels_df["index"], atlas_labels_df["label"])
    orig_str = ""
    mrtrix_str = ""
    for i_row, (index, label) in enumerate(index_label_pairs):
        # TODO: Consider using special delimiters that can be replaced with spaces before output
        # files are written.
        orig_str += f"{index}\t{label.replace(' ', '-')}\n"
        mrtrix_str += f"{i_row + 1}\t{label.replace(' ', '-')}\n"

    with open(mrtrix_txt, "w") as mrtrix_f:
        mrtrix_f.write(mrtrix_str)

    with open(orig_txt, "w") as orig_f:
        orig_f.write(orig_str)

    cmd = ["labelconvert", original_atlas, orig_txt, mrtrix_txt, output_mif]
    subprocess.run(cmd, check=True)
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


class _SplitAtlasConfigsInputSpec(BaseInterfaceInputSpec):
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


class _SplitAtlasConfigsOutputSpec(TraitedSpec):
    atlas_configs = traits.List(
        traits.Dict(),
        desc=(
            "Dictionary of atlas configurations. "
            "This interface adds the following keys: "
            "'dwi_resolution_file', 'dwi_resolution_mif', 'orig_lut', 'mrtrix_lut'. "
            "The values are the paths to the transformed atlas files and the label files."
        ),
    )


class SplitAtlasConfigs(SimpleInterface):
    input_spec = _SplitAtlasConfigsInputSpec
    output_spec = _SplitAtlasConfigsOutputSpec

    def _run_interface(self, runtime):
        atlas_configs = []
        for atlas_name, atlas_config in self.inputs.atlas_configs.items():
            atlas_configs.append({atlas_name: atlas_config})

        self._results["atlas_configs"] = atlas_configs


class _ExtractAtlasFilesInputSpec(BaseInterfaceInputSpec):
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


class _ExtractAtlasFilesOutputSpec(TraitedSpec):
    atlases = traits.List(traits.Str(), desc="List of atlas names")
    nifti_files = traits.List(File(), desc="List of nifti files")
    mif_files = traits.List(File(), desc="List of mif files")
    mrtrix_lut_files = traits.List(File(), desc="List of mrtrix lut files")
    orig_lut_files = traits.List(File(), desc="List of orig lut files")


class ExtractAtlasFiles(SimpleInterface):
    input_spec = _ExtractAtlasFilesInputSpec
    output_spec = _ExtractAtlasFilesOutputSpec

    def _run_interface(self, runtime):
        self._results["atlases"] = []
        self._results["nifti_files"] = []
        self._results["mif_files"] = []
        self._results["mrtrix_lut_files"] = []
        self._results["orig_lut_files"] = []

        for atlas_name, atlas_config in self.inputs.atlas_configs.items():
            self._results["atlases"].append(atlas_name)
            self._results["nifti_files"].append(atlas_config["dwi_resolution_file"])
            self._results["mif_files"].append(atlas_config["dwi_resolution_mif"])
            self._results["mrtrix_lut_files"].append(atlas_config["mrtrix_lut"])
            self._results["orig_lut_files"].append(atlas_config["orig_lut"])

        return runtime


class _RecombineAtlasConfigsInputSpec(BaseInterfaceInputSpec):
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
    atlases = traits.List(traits.Str(), desc="List of atlas names")
    nifti_files = traits.List(File(), desc="List of nifti files")
    mif_files = traits.List(File(), desc="List of mif files")
    mrtrix_lut_files = traits.List(File(), desc="List of mrtrix lut files")
    orig_lut_files = traits.List(File(), desc="List of orig lut files")


class _RecombineAtlasConfigsOutputSpec(TraitedSpec):
    atlas_configs = traits.Dict(
        mandatory=True,
        desc=(
            "Dictionary of atlas configurations. "
            "Keys are atlas names and values are dictionaries with the following keys: "
            "'file', 'label', 'metadata'. "
        ),
    )


class RecombineAtlasConfigs(SimpleInterface):
    input_spec = _RecombineAtlasConfigsInputSpec
    output_spec = _RecombineAtlasConfigsOutputSpec

    def _run_interface(self, runtime):
        atlas_configs = self.inputs.atlas_configs.copy()

        for i_atlas, atlas_name in enumerate(self.inputs.atlases):
            atlas_configs[atlas_name]["dwi_resolution_file"] = self.inputs.nifti_files[i_atlas]
            atlas_configs[atlas_name]["dwi_resolution_mif"] = self.inputs.mif_files[i_atlas]
            atlas_configs[atlas_name]["mrtrix_lut"] = self.inputs.mrtrix_lut_files[i_atlas]
            atlas_configs[atlas_name]["orig_lut"] = self.inputs.orig_lut_files[i_atlas]

        self._results["atlas_configs"] = atlas_configs

        return runtime
