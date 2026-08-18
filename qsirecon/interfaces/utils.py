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
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.interfaces.header import ValidateImage

IFLOGGER = logging.getLogger('nipype.interface')


class _AtlasLUTsInputSpec(BaseInterfaceInputSpec):
    atlas_labels_file = File(
        exists=True,
        mandatory=True,
        desc=(
            'Atlas labels file (tsv) to read in. '
            'This file should have at least two columns: index and name.'
        ),
    )


class _AtlasLUTsOutputSpec(TraitedSpec):
    orig_lut = File(exists=True, desc="Lookup table with the atlas's original indices.")
    mrtrix_lut = File(exists=True, desc='Lookup table with sequential indices, for MRtrix.')


class AtlasLUTs(SimpleInterface):
    """Write MRtrix-format lookup tables for an atlas.

    ``labelconvert`` uses these two tables to renumber an atlas's original indices
    into the sequential indices (1 to N) that MRtrix's connectivity tools expect.
    """

    input_spec = _AtlasLUTsInputSpec
    output_spec = _AtlasLUTsOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        orig_lut = fname_presuffix(
            self.inputs.atlas_labels_file,
            newpath=runtime.cwd,
            suffix='_origlabels.txt',
            use_ext=False,
        )
        mrtrix_lut = fname_presuffix(
            self.inputs.atlas_labels_file,
            newpath=runtime.cwd,
            suffix='_mrtrixlabels.txt',
            use_ext=False,
        )

        atlas_labels_df = pd.read_table(self.inputs.atlas_labels_file)
        atlas_labels_df['index'] = atlas_labels_df['index'].astype(int)
        if 0 in atlas_labels_df['index'].values:
            IFLOGGER.warning(
                f'Atlas {self.inputs.atlas_labels_file} has a 0 index. This index will be dropped.'
            )
            atlas_labels_df = atlas_labels_df.loc[atlas_labels_df['index'] != 0]

        orig_str = ''
        mrtrix_str = ''
        index_label_pairs = zip(atlas_labels_df['index'], atlas_labels_df['name'])
        for i_row, (index, label) in enumerate(index_label_pairs):
            # TODO: Consider using special delimiters that can be replaced with spaces before
            # output files are written.
            orig_str += f'{index}\t{label.replace(" ", "-")}\n'
            mrtrix_str += f'{i_row + 1}\t{label.replace(" ", "-")}\n'

        with open(orig_lut, 'w') as orig_f:
            orig_f.write(orig_str)

        with open(mrtrix_lut, 'w') as mrtrix_f:
            mrtrix_f.write(mrtrix_str)

        self._results['orig_lut'] = orig_lut
        self._results['mrtrix_lut'] = mrtrix_lut

        return runtime


class _ConformAtlasInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='dwi image')
    orientation = traits.Enum('LPS', 'LAS', default='LPS', usedefault=True)


class _ConformAtlasOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='conformed dwi image')


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
        suffix = '_' + orientation
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
            self._results['out_file'] = out_fname

        else:
            self._results['out_file'] = fname

        return runtime


class _WriteSidecarInputSpec(BaseInterfaceInputSpec):
    metadata = traits.Dict()


class _WriteSidecarOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class WriteSidecar(SimpleInterface):
    input_spec = _WriteSidecarInputSpec
    output_spec = _WriteSidecarOutputSpec

    def _run_interface(self, runtime):
        out_file = os.path.join(runtime.cwd, 'sidecar.json')
        with open(out_file, 'w') as outf:
            json.dump(self.inputs.metadata, outf)
        self._results['out_file'] = out_file
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
        out_file = os.path.join(runtime.cwd, 'brainfig.png')
        plotting.plot_img(
            img=img, output_file=out_file, title=os.path.basename(self.inputs.dwi_file)
        )
        self._results['out_file'] = out_file
        return runtime


class _SplitAtlasConfigsInputSpec(BaseInterfaceInputSpec):
    atlas_configs = traits.Dict(
        mandatory=True,
        desc=(
            'Dictionary of atlas configurations. '
            'Keys are atlas names and values are dictionaries with the following keys: '
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
            'Dictionary of atlas configurations. '
            'This interface adds the following keys: '
            "'dwi_resolution_file', 'dwi_resolution_mif', 'orig_lut', 'mrtrix_lut'. "
            'The values are the paths to the transformed atlas files and the label files.'
        ),
    )


class SplitAtlasConfigs(SimpleInterface):
    input_spec = _SplitAtlasConfigsInputSpec
    output_spec = _SplitAtlasConfigsOutputSpec

    def _run_interface(self, runtime):
        atlas_configs = []
        for atlas_name, atlas_config in self.inputs.atlas_configs.items():
            atlas_configs.append({atlas_name: atlas_config})

        self._results['atlas_configs'] = atlas_configs

        return runtime


class _RecombineAtlasConfigsInputSpec(BaseInterfaceInputSpec):
    atlas_configs = traits.Dict(
        mandatory=True,
        desc=(
            'Dictionary of atlas configurations. '
            'Keys are atlas names and values are dictionaries with the following keys: '
            "'file', 'label', 'metadata'. "
            "'file' is the path to the atlas file. "
            "'label' is the path to the label file. "
            "'metadata' is a dictionary with relevant metadata. "
            "'xfm_to_anat' is the path to the transform to get the atlas into ACPC space."
        ),
    )
    atlases = traits.List(traits.Str(), desc='List of atlas names')
    nifti_files = traits.List(File(), desc='List of nifti files')
    mif_files = traits.List(File(), desc='List of mif files')
    mrtrix_lut_files = traits.List(File(), desc='List of mrtrix lut files')
    orig_lut_files = traits.List(File(), desc='List of orig lut files')


class _RecombineAtlasConfigsOutputSpec(TraitedSpec):
    atlas_configs = traits.Dict(
        mandatory=True,
        desc=(
            'Dictionary of atlas configurations. '
            'Keys are atlas names and values are dictionaries with the following keys: '
            "'file', 'label', 'metadata'. "
        ),
    )


class RecombineAtlasConfigs(SimpleInterface):
    input_spec = _RecombineAtlasConfigsInputSpec
    output_spec = _RecombineAtlasConfigsOutputSpec

    def _run_interface(self, runtime):
        atlas_configs = self.inputs.atlas_configs.copy()

        for i_atlas, atlas_name in enumerate(self.inputs.atlases):
            atlas_configs[atlas_name]['dwi_resolution_file'] = self.inputs.nifti_files[i_atlas]
            atlas_configs[atlas_name]['dwi_resolution_mif'] = self.inputs.mif_files[i_atlas]
            atlas_configs[atlas_name]['mrtrix_lut'] = self.inputs.mrtrix_lut_files[i_atlas]
            atlas_configs[atlas_name]['orig_lut'] = self.inputs.orig_lut_files[i_atlas]

        self._results['atlas_configs'] = atlas_configs

        return runtime


class _LoadResponseFunctionsInputSpec(BaseInterfaceInputSpec):
    wm_file = File(
        exists=False,
        mandatory=True,
        desc='WM response function file. Only MRtrix-format txt files are currently supported.',
    )
    gm_file = traits.Either(
        None,
        File(
            exists=False,
            mandatory=False,
            desc=(
                'GM response function file. Only MRtrix-format txt files are currently supported.'
            ),
        ),
    )
    csf_file = traits.Either(
        None,
        File(
            exists=False,
            mandatory=False,
            desc=(
                'CSF response function file. Only MRtrix-format txt files are currently supported.'
            ),
        ),
    )
    using_multitissue = traits.Bool(desc='Whether to use multitissue response functions or not.')
    input_dir = traits.Directory(
        exists=True,
        mandatory=True,
        desc='Directory containing response function files.',
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
        self._results['wm_txt'] = wm_file
        if not os.path.exists(wm_file):
            raise FileNotFoundError(f'WM response file {wm_file} not found')

        if self.inputs.gm_file and self.inputs.using_multitissue:
            gm_file = os.path.abspath(os.path.join(self.inputs.input_dir, self.inputs.gm_file))
            if not os.path.exists(gm_file):
                raise FileNotFoundError(f'GM response file {gm_file} not found')
            self._results['gm_txt'] = gm_file
        elif self.inputs.using_multitissue:
            raise ValueError('gm_file is required when using multitissue response functions')

        if self.inputs.csf_file and self.inputs.using_multitissue:
            csf_file = os.path.abspath(os.path.join(self.inputs.input_dir, self.inputs.csf_file))
            if not os.path.exists(csf_file):
                raise FileNotFoundError(f'CSF response file {csf_file} not found')
            self._results['csf_txt'] = csf_file
        elif self.inputs.using_multitissue:
            raise ValueError('csf_file is required when using multitissue response functions')

        return runtime
