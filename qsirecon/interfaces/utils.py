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
            "'dwi_resolution_niigz', 'dwi_resolution_mif', 'mrtrix_lut'. "
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
            output_nii = fname_presuffix(
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

            atlas_configs[atlas_name]["dwi_resolution_niigz"] = output_nii
            atlas_configs[atlas_name]["dwi_resolution_mif"] = output_mif
            atlas_configs[atlas_name]["orig_lut"] = output_orig_txt
            atlas_configs[atlas_name]["mrtrix_lut"] = output_mif_txt

            conform_atlas = ConformAtlas(in_file=atlas_config["image"], orientation="LPS")
            result = conform_atlas.run()
            lps_name = result.outputs.out_file
            resampled_nii = fname_presuffix(
                lps_name,
                newpath=runtime.cwd,
                suffix="_resampled.nii.gz",
            )

            resample_commands.append(
                _resample_atlas(
                    input_atlas=lps_name,
                    output_atlas=resampled_nii,
                    transform=transforms[i_atlas],
                    ref_image=self.inputs.reference_image,
                )
            )
            label_convert(
                original_atlas=atlas_config["image"],
                output_mif=output_mif,
                orig_txt=output_orig_txt,
                mrtrix_txt=output_mif_txt,
                atlas_labels_file=atlas_config["labels"],
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


def label_convert(
    original_atlas, converted_mif, converted_nii, orig_txt, mrtrix_txt, atlas_labels_file
):
    """Create a mrtrix label file from an atlas.

    Parameters
    ----------
    original_atlas : str
        Path to the original atlas file, in NIfTI format.
    converted_mif : str
        Path to the output mrtrix label file (mif[.gz]) to be written out.
    converted_nii : str
        Path to the output NIfTI label file (nii[.gz]) to be written out.
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
        print(f"WARNING: Atlas {atlas_labels_file} has a 0 index. This index will be dropped.")
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

    # First run labelconvert to create a MIF file.
    cmd = ["labelconvert", original_atlas, orig_txt, mrtrix_txt, converted_mif]
    subprocess.run(cmd, check=True)
    if not os.path.isfile(converted_mif):
        raise RuntimeError(
            f"Failed to create mrtrix label file ({converted_mif}) from {original_atlas}"
        )

    # Then run labelconvert to create a NIfTI file.
    cmd = ["labelconvert", original_atlas, orig_txt, mrtrix_txt, converted_nii]
    subprocess.run(cmd, check=True)
    if not os.path.isfile(converted_nii):
        raise RuntimeError(
            f"Failed to create NIfTI label file ({converted_nii}) from {original_atlas}"
        )


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


class _DeconstructAtlasConfigsInputSpec(BaseInterfaceInputSpec):
    atlas_configs = traits.Dict(
        mandatory=True,
        desc=(
            "Dictionary of atlas configurations. "
            "Keys are atlas names and values are dictionaries with the following keys: "
            "'file', 'label', 'metadata'. "
            "'file' is the path to the atlas file. "
            "'label' is the path to the label file. "
            "'metadata' is a dictionary with relevant metadata. "
            "'xfm_to_anat' is the path to the transform to get the atlas into ACPC space."
        ),
    )


class _DeconstructAtlasConfigsOutputSpec(TraitedSpec):
    atlases_names = traits.List(traits.Str(), desc="List of atlas names")
    nifti_files = traits.List(File(), desc="List of nifti files")
    atlas_labels_files = traits.List(File(), desc="List of atlas labels files")
    xfms_to_anat = traits.List(File(), desc="List of transforms to get the atlas into ACPC space")


class DeconstructAtlasConfigs(SimpleInterface):
    input_spec = _DeconstructAtlasConfigsInputSpec
    output_spec = _DeconstructAtlasConfigsOutputSpec

    def _run_interface(self, runtime):
        self._results["atlases_names"] = []
        self._results["nifti_files"] = []
        self._results["atlas_labels_files"] = []
        self._results["xfms_to_anat"] = []

        for atlas_name, atlas_config in self.inputs.atlas_configs.items():
            self._results["atlases_names"].append(atlas_name)
            self._results["nifti_files"].append(atlas_config["dwi_resolution_niigz"])
            self._results["atlas_labels_files"].append(atlas_config["labels"])
            self._results["xfms_to_anat"].append(atlas_config["xfm_to_anat"])

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
            "'xfm_to_anat' is the path to the transform to get the atlas into ACPC space."
        ),
    )


class _SplitAtlasConfigsOutputSpec(TraitedSpec):
    atlas_configs = traits.List(
        traits.Dict(),
        desc=(
            "Dictionary of atlas configurations. "
            "This interface adds the following keys: "
            "'dwi_resolution_niigz', 'dwi_resolution_mif', 'mrtrix_lut'. "
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
            "'xfm_to_anat' is the path to the transform to get the atlas into ACPC space."
        ),
    )


class _ExtractAtlasFilesOutputSpec(TraitedSpec):
    atlases = traits.List(traits.Str(), desc="List of atlas names")
    nifti_files = traits.List(File(), desc="List of nifti files")
    mif_files = traits.List(File(), desc="List of mif files")
    mrtrix_lut_files = traits.List(File(), desc="List of mrtrix lut files")
    # XXX: Need the following
    xfms_to_anat = traits.List(File(), desc="List of transforms to get the atlas into ACPC space")
    labels_files = traits.List(File(), desc="List of labels files")


class ExtractAtlasFiles(SimpleInterface):
    input_spec = _ExtractAtlasFilesInputSpec
    output_spec = _ExtractAtlasFilesOutputSpec

    def _run_interface(self, runtime):
        self._results["atlases"] = []
        self._results["labels_files"] = []
        self._results["nifti_files"] = []
        self._results["mif_files"] = []
        self._results["mrtrix_lut_files"] = []
        self._results["xfms_to_anat"] = []

        for atlas_name, atlas_config in self.inputs.atlas_configs.items():
            self._results["atlases"].append(atlas_name)
            self._results["labels_files"].append(atlas_config.get("labels", None))
            self._results["nifti_files"].append(atlas_config.get("dwi_resolution_niigz", None))
            self._results["mif_files"].append(atlas_config.get("dwi_resolution_mif", None))
            self._results["mrtrix_lut_files"].append(atlas_config.get("mrtrix_lut", None))
            self._results["xfms_to_anat"].append(atlas_config.get("xfm_to_anat", None))

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
            "'xfm_to_anat' is the path to the transform to get the atlas into ACPC space."
        ),
    )
    atlases = traits.List(traits.Str(), desc="List of atlas names")
    nifti_files = traits.List(File(), desc="List of nifti files")
    mif_files = traits.List(File(), desc="List of mif files")
    mrtrix_lut_files = traits.List(File(), desc="List of mrtrix lut files")


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
        from nipype.interfaces.base import isdefined

        atlas_configs = {}
        if isdefined(self.inputs.atlas_configs):
            atlas_configs = self.inputs.atlas_configs.copy()

        for i_atlas, atlas_name in enumerate(self.inputs.atlases):
            if isdefined(self.inputs.nifti_files):
                atlas_configs[atlas_name]["dwi_resolution_niigz"] = (
                    self.inputs.nifti_files[i_atlas]
                )
            if isdefined(self.inputs.mif_files):
                atlas_configs[atlas_name]["dwi_resolution_mif"] = self.inputs.mif_files[i_atlas]
            if isdefined(self.inputs.mrtrix_lut_files):
                atlas_configs[atlas_name]["mrtrix_lut"] = self.inputs.mrtrix_lut_files[i_atlas]

        self._results["atlas_configs"] = atlas_configs

        return runtime


class _LoadResponseFunctionsInputSpec(BaseInterfaceInputSpec):
    wm_file = File(
        exists=False,
        mandatory=True,
        desc="WM response function file. Only MRtrix-format txt files are currently supported.",
    )
    gm_file = traits.Either(
        None,
        File(
            exists=False,
            mandatory=False,
            desc=(
                "GM response function file. Only MRtrix-format txt files are currently "
                "supported."
            ),
        ),
    )
    csf_file = traits.Either(
        None,
        File(
            exists=False,
            mandatory=False,
            desc=(
                "CSF response function file. Only MRtrix-format txt files are currently "
                "supported."
            ),
        ),
    )
    using_multitissue = traits.Bool(desc="Whether to use multitissue response functions or not.")
    input_dir = traits.Directory(
        exists=True,
        mandatory=True,
        desc="Directory containing response function files.",
    )


class _LoadResponseFunctionsOutputSpec(TraitedSpec):
    wm_txt = File(exists=True)
    gm_txt = File(exists=True)
    csf_txt = File(exists=True)


class LoadResponseFunctions(SimpleInterface):
    """Collect response function files from the input directory.

    The names of the response function files are specified in the reconstruction specification,
    and must be located in the recon_spec_aux_files directory.

    TODO: Support BEP016-format JSON files.
    """

    input_spec = _LoadResponseFunctionsInputSpec
    output_spec = _LoadResponseFunctionsOutputSpec

    def _run_interface(self, runtime):
        wm_file = os.path.abspath(os.path.join(self.inputs.input_dir, self.inputs.wm_file))
        self._results["wm_txt"] = wm_file
        if not os.path.exists(wm_file):
            raise FileNotFoundError(f"WM response file {wm_file} not found")

        if self.inputs.gm_file and self.inputs.using_multitissue:
            gm_file = os.path.abspath(os.path.join(self.inputs.input_dir, self.inputs.gm_file))
            if not os.path.exists(gm_file):
                raise FileNotFoundError(f"GM response file {gm_file} not found")
            self._results["gm_txt"] = gm_file
        elif self.inputs.using_multitissue:
            raise ValueError("gm_file is required when using multitissue response functions")

        if self.inputs.csf_file and self.inputs.using_multitissue:
            csf_file = os.path.abspath(os.path.join(self.inputs.input_dir, self.inputs.csf_file))
            if not os.path.exists(csf_file):
                raise FileNotFoundError(f"CSF response file {csf_file} not found")
            self._results["csf_txt"] = csf_file
        elif self.inputs.using_multitissue:
            raise ValueError("csf_file is required when using multitissue response functions")

        return runtime


class _MakeLUTsInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc="atlas labels file")


class _MakeLUTsOutputSpec(TraitedSpec):
    orig_lut = File(exists=True, desc="orig lut file")
    mrtrix_lut = File(exists=True, desc="mrtrix lut file")


class MakeLUTs(SimpleInterface):
    input_spec = _MakeLUTsInputSpec
    output_spec = _MakeLUTsOutputSpec

    def _run_interface(self, runtime):
        self._results["orig_lut"] = self.inputs.in_file
        self._results["mrtrix_lut"] = self.inputs.in_file
        return runtime
