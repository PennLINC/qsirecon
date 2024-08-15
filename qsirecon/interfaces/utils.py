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

from ..utils.atlases import get_atlases

IFLOGGER = logging.getLogger("nipype.interfaces")


class GetConnectivityAtlasesInputSpec(BaseInterfaceInputSpec):
    atlas_names = traits.List(mandatory=True, desc="atlas names to be used")
    forward_transform = File(exists=True, desc="transform to get atlas into T1w space if desired")
    reference_image = File(exists=True, desc="")
    space = traits.Str("T1w", usedefault=True)


class GetConnectivityAtlasesOutputSpec(TraitedSpec):
    atlas_configs = traits.Dict()
    commands = File()


class GetConnectivityAtlases(SimpleInterface):
    input_spec = GetConnectivityAtlasesInputSpec
    output_spec = GetConnectivityAtlasesOutputSpec

    def _run_interface(self, runtime):
        atlas_names = self.inputs.atlas_names
        atlas_configs = get_atlases(atlas_names)

        if self.inputs.space == "T1w":
            if not isdefined(self.inputs.forward_transform):
                raise Exception("No MNI to T1w transform found in anatomical directory")
            else:
                transform = self.inputs.forward_transform
        else:
            transform = "identity"

        # Transform atlases to match the DWI data
        resample_commands = []
        for atlas_name, atlas_config in atlas_configs.items():
            output_name = fname_presuffix(
                atlas_config["file"], newpath=runtime.cwd, suffix="_to_dwi"
            )
            output_mif = fname_presuffix(
                atlas_config["file"], newpath=runtime.cwd, suffix="_to_dwi.mif", use_ext=False
            )
            output_mif_txt = fname_presuffix(
                atlas_config["file"],
                newpath=runtime.cwd,
                suffix="_mrtrixlabels.txt",
                use_ext=False,
            )
            output_orig_txt = fname_presuffix(
                atlas_config["file"], newpath=runtime.cwd, suffix="_origlabels.txt", use_ext=False
            )

            atlas_configs[atlas_name]["dwi_resolution_file"] = output_name
            atlas_configs[atlas_name]["dwi_resolution_mif"] = output_mif
            atlas_configs[atlas_name]["orig_lut"] = output_mif_txt
            atlas_configs[atlas_name]["mrtrix_lut"] = output_orig_txt
            resample_commands.append(
                _resample_atlas(
                    input_atlas=atlas_config["file"],
                    output_atlas=output_name,
                    transform=transform,
                    ref_image=self.inputs.reference_image,
                )
            )
            label_convert(output_name, output_mif, output_orig_txt, output_mif_txt, atlas_config)

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


def label_convert(original_atlas, output_mif, orig_txt, mrtrix_txt, metadata):
    """Create a mrtrix label file from an atlas."""

    with open(mrtrix_txt, "w") as mrtrix_f:
        with open(orig_txt, "w") as orig_f:
            for row_num, (roi_num, roi_name) in enumerate(
                zip(metadata["node_ids"], metadata["node_names"])
            ):
                orig_f.write("{}\t{}\n".format(roi_num, roi_name))
                mrtrix_f.write("{}\t{}\n".format(row_num + 1, roi_name))
    cmd = ["labelconvert", original_atlas, orig_txt, mrtrix_txt, output_mif]
    os.system(" ".join(cmd))


class _PNGtoSVGInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="PNG to convert to SVG.",
    )


class _PNGtoSVGOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Converted SVG file.",
    )


class PNGtoSVG(SimpleInterface):
    """Collect registration files for fsnative-to-fsLR transformation."""

    input_spec = _PNGtoSVGInputSpec
    output_spec = _PNGtoSVGOutputSpec

    def _run_interface(self, runtime):
        from PIL import Image
        import svgwrite

        # Open the PNG file
        in_file = self.inputs.in_file
        png_image = Image.open(in_file)

        # Get the dimensions of the PNG image
        width, height = png_image.size

        # Create an SVG drawing
        out_file = os.path.abspath("output.svg")
        dwg = svgwrite.Drawing(out_file, profile="tiny", size=(width, height))

        # Embed the PNG image within the SVG
        dwg.add(dwg.image(in_file, insert=(0, 0), size=(width, height)))

        # Save the SVG file
        dwg.save()

        self._results["out_file"] = out_file

        return runtime
