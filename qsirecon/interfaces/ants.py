#!python
import logging
import os.path as op

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import split_filename

LOGGER = logging.getLogger("nipype.interface")


class ImageMathInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, position=3, argstr="%s")
    dimension = traits.Enum(3, 2, 4, usedefault=True, argstr="%d", position=0)
    out_file = File(argstr="%s", genfile=True, position=1)
    operation = traits.Str(argstr="%s", position=2)
    secondary_arg = traits.Str("", argstr="%s")
    secondary_file = File(argstr="%s")


class ImageMathOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ImageMath(CommandLine):
    input_spec = ImageMathInputSpec
    output_spec = ImageMathOutputSpec
    _cmd = "ImageMath"

    def _gen_filename(self, name):
        if name == "out_file":
            output = self.inputs.out_file
            if not isdefined(output):
                _, fname, ext = split_filename(self.inputs.in_file)
                output = fname + "_" + self.inputs.operation + ext
            return output
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self._gen_filename("out_file"))
        return outputs


class _ConvertTransformFileInputSpec(CommandLineInputSpec):
    dimension = traits.Enum((3, 2), default=3, usedefault=True, argstr="%d", position=0)
    in_transform = traits.File(exists=True, argstr="%s", mandatory=True, position=1)
    out_transform = traits.File(
        argstr="%s",
        name_source="in_transform",
        name_template="%s.txt",
        keep_extension=False,
        position=2,
    )


class _ConvertTransformFileOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ConvertTransformFile(CommandLine):
    _cmd = "ConvertTransformFile"
    input_spec = _ConvertTransformFileInputSpec
    output_spec = _ConvertTransformFileOutputSpec
