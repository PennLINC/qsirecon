#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utilities
^^^^^^^^^^^^^^^^^^^^^^^

"""

import os

from nipype import logging
from nipype.interfaces import ants
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

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
            resample_commands.append(
                _resample_atlas(
                    input_atlas=atlas_config["image"],
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


class _GetUniqueInputSpec(BaseInterfaceInputSpec):
    inlist = traits.List(mandatory=True, desc="list of things")


class _GetUniqueOutputSpec(TraitedSpec):
    outlist = traits.List()


class GetUnique(SimpleInterface):
    input_spec = _GetUniqueInputSpec
    output_spec = _GetUniqueOutputSpec

    def _run_interface(self, runtime):
        in_list = self.inputs.inlist
        in_list = [x for x in in_list if isdefined(x)]
        self._results["outlist"] = sorted(list(set(in_list)))
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
    for index, label in index_label_pairs:
        orig_str += f"{index}\t{label}\n"
        mrtrix_str += f"{index + 1}\t{label}\n"

    with open(mrtrix_txt, "w") as mrtrix_f:
        mrtrix_f.write(mrtrix_str)

    with open(orig_txt, "w") as orig_f:
        orig_f.write(orig_str)

    cmd = ["labelconvert", original_atlas, orig_txt, mrtrix_txt, output_mif]
    os.system(" ".join(cmd))
