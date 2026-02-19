# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""

import gzip
import os
import os.path as op
import re
from json import dump, loads
from shutil import copyfile, copyfileobj, copytree

import nibabel as nb
import numpy as np
from bids.layout import Config, parse_file_entities
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import split_filename
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink
from niworkflows.interfaces.bids import (
    _DerivativesDataSinkInputSpec,
    _DerivativesDataSinkOutputSpec,
)

from qsirecon import config
from qsirecon.data import load as load_data

LOGGER = logging.getLogger('nipype.interface')
BIDS_NAME = re.compile(
    r'^(.*\/)?(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
    '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
    '(_(?P<space_id>space-[a-zA-Z0-9]+))?'
    '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?'
)

# NOTE: Modified for QSIRecon's purposes
qsirecon_spec = loads(load_data('io_spec.json').read_text())
atlas_spec = loads(load_data('atlas_bids_config.json').read_text())
bids_config = Config.load('bids')
deriv_config = Config.load('derivatives')

qsirecon_entities = {v['name']: v['pattern'] for v in qsirecon_spec['entities']}
atlas_entities = {v['name']: v['pattern'] for v in atlas_spec['entities']}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **qsirecon_entities, **atlas_entities}
merged_entities = [{'name': k, 'pattern': v} for k, v in merged_entities.items()]
config_entities = frozenset({e['name'] for e in merged_entities})


def get_bids_params(fullpath):
    bids_patterns = [
        r'^(.*/)?(?P<subject_id>sub-[a-zA-Z0-9]+)',
        '^.*_(?P<session_id>ses-[a-zA-Z0-9]+)',
        '^.*_(?P<task_id>task-[a-zA-Z0-9]+)',
        '^.*_(?P<acq_id>acq-[a-zA-Z0-9]+)',
        '^.*_(?P<space_id>space-[a-zA-Z0-9]+)',
        '^.*_(?P<rec_id>rec-[a-zA-Z0-9]+)',
        '^.*_(?P<run_id>run-[a-zA-Z0-9]+)',
        '^.*_(?P<dir_id>dir-[a-zA-Z0-9]+)',
    ]
    matches = {
        'subject_id': None,
        'session_id': None,
        'task_id': None,
        'dir_id': None,
        'acq_id': None,
        'space_id': None,
        'rec_id': None,
        'run_id': None,
    }
    for pattern in bids_patterns:
        pat = re.compile(pattern)
        match = pat.search(fullpath)
        params = match.groupdict() if match is not None else {}
        matches.update(params)
    return matches


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using QSIRecon's configuration files.
    """

    out_path_base = ''
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = qsirecon_spec['default_path_patterns']


class _CopyAtlasInputSpec(BaseInterfaceInputSpec):
    source_file = traits.Str(
        desc="The source file's name.",
        mandatory=False,
    )
    in_file = File(
        exists=True,
        desc='The atlas file to copy.',
        mandatory=True,
    )
    meta_dict = traits.Either(
        traits.Dict(),
        None,
        desc='The atlas metadata dictionary.',
        mandatory=False,
    )
    out_dir = Directory(
        exists=True,
        desc='The output directory.',
        mandatory=True,
    )
    atlas = traits.Str(
        desc='The atlas name.',
        mandatory=True,
    )
    Sources = traits.List(
        traits.Str,
        desc='List of sources for the atlas.',
        mandatory=False,
    )


class _CopyAtlasOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='The copied atlas file.',
    )


class CopyAtlas(SimpleInterface):
    """Copy atlas file to output directory.

    Parameters
    ----------
    source_file : :obj:`str`
        The source name of the atlas file.
    in_file : :obj:`str`
        The atlas file to copy.
    out_dir : :obj:`str`
        The output directory.
    atlas : :obj:`str`
        The name of the atlas.

    Returns
    -------
    out_file : :obj:`str`
        The path to the copied atlas file.

    Notes
    -----
    I can't use DerivativesDataSink because it has a problem with dlabel CIFTI files.
    It gives the following error:
    "AttributeError: 'Cifti2Header' object has no attribute 'set_data_dtype'"

    I can't override the CIFTI atlas's data dtype ahead of time because setting it to int8 or int16
    somehow converts all of the values in the data array to weird floats.
    This could be a version-specific nibabel issue.

    I've also updated this function to handle JSON and TSV files as well.
    """

    input_spec = _CopyAtlasInputSpec
    output_spec = _CopyAtlasOutputSpec

    def _run_interface(self, runtime):
        out_dir = self.inputs.out_dir
        in_file = self.inputs.in_file
        meta_dict = self.inputs.meta_dict
        source_file = self.inputs.source_file
        atlas = self.inputs.atlas
        Sources = self.inputs.Sources

        atlas_out_dir = os.path.join(out_dir, f'atlases/atlas-{atlas}')

        if in_file.endswith('.tsv'):
            out_basename = f'atlas-{atlas}_dseg'
            extension = '.tsv'
        else:
            extension = '.nii.gz' if source_file.endswith('.nii.gz') else '.dlabel.nii'
            space = get_entity(source_file, 'space')
            res = get_entity(source_file, 'res')
            den = get_entity(source_file, 'den')
            cohort = get_entity(source_file, 'cohort')

            cohort_str = f'_cohort-{cohort}' if cohort else ''
            res_str = f'_res-{res}' if res else ''
            den_str = f'_den-{den}' if den else ''
            if extension == '.dlabel.nii':
                out_basename = f'atlas-{atlas}_space-{space}{den_str}{cohort_str}_dseg'
            elif extension == '.nii.gz':
                out_basename = f'atlas-{atlas}_space-{space}{res_str}{cohort_str}_dseg'

        os.makedirs(atlas_out_dir, exist_ok=True)
        out_file = os.path.join(atlas_out_dir, f'{out_basename}{extension}')

        if out_file.endswith('.nii.gz') and os.path.isfile(out_file):
            # Check that native-resolution atlas doesn't have a different resolution from the last
            # run's atlas.
            old_img = nb.load(out_file)
            new_img = nb.load(in_file)
            if not np.allclose(old_img.affine, new_img.affine):
                raise ValueError(
                    f"Existing '{atlas}' atlas affine ({out_file}) is different from the input "
                    f'file affine ({in_file}).'
                )

        # Don't copy the file if it exists, to prevent any race conditions between parallel
        # processes.
        if not os.path.isfile(out_file):
            copyfile(in_file, out_file)

        # Only write out a sidecar if metadata are provided
        if meta_dict or Sources:
            meta_file = os.path.join(atlas_out_dir, f'{out_basename}.json')
            meta_dict = meta_dict or {}
            meta_dict = meta_dict.copy()
            if Sources:
                meta_dict['Sources'] = meta_dict.get('Sources', []) + Sources

            with open(meta_file, 'w') as fo:
                dump(meta_dict, fo, sort_keys=True, indent=4)

        self._results['out_file'] = out_file

        return runtime


def get_recon_output_name(
    base_dir,
    source_file,
    derivative_file,
    output_bids_entities,
    use_ext=True,
    qsirecon_suffix=None,
    dismiss_entities=None,
):
    source_entities = parse_file_entities(source_file)
    if dismiss_entities:
        source_entities = {k: v for k, v in source_entities.items() if k not in dismiss_entities}

    out_path = base_dir
    if qsirecon_suffix and qsirecon_suffix.lower() != 'qsirecon':
        out_path = op.join(out_path, 'derivatives', f'qsirecon-{qsirecon_suffix}')

    # Infer the appropriate extension
    if 'extension' not in output_bids_entities:
        ext_parts = os.path.basename(derivative_file).split('.')[1:]
        if len(ext_parts) > 2:
            ext = split_filename(derivative_file)[2]
        else:
            ext = '.' + '.'.join(ext_parts)

        output_bids_entities['extension'] = ext

    # Add the suffix
    output_bids_entities['suffix'] = output_bids_entities.get('suffix', 'dwimap')

    # Add any missing entities from the source file
    output_bids_entities = {**source_entities, **output_bids_entities}

    out_filename = config.execution.layout.build_path(
        source=output_bids_entities,
        path_patterns=qsirecon_spec['default_path_patterns'],
        validate=False,
        absolute_paths=False,
    )
    if not use_ext:
        # Drop the extension from the filename
        out_filename = out_filename.split('.')[0]

    return os.path.join(out_path, out_filename)


class _ReconDerivativesDataSinkInputSpec(_DerivativesDataSinkInputSpec):
    in_file = traits.Either(
        traits.Directory(exists=True),
        InputMultiObject(File(exists=True)),
        mandatory=True,
        desc='the object to be saved',
    )
    param = traits.Str('', usedefault=True, desc='Label for parameter field')
    model = traits.Str('', usedefault=True, desc='Label for model field')
    bundle = traits.Str('', usedefault=True, desc='Label for bundle field')
    bundles = traits.Str('', usedefault=True, desc='Label for bundles field')
    label = traits.Str('', usedefault=True, desc='Label for label field')
    atlas = traits.Str('', usedefault=True, desc='Label for label field')
    extension = traits.Str('', usedefault=True, desc='Extension (will be ignored)')
    qsirecon_suffix = traits.Str(
        '', usedefault=True, desc='name appended to qsirecon- in the derivatives'
    )


class _ReconDerivativesDataSinkOutputSpec(_DerivativesDataSinkOutputSpec):
    out_file = traits.Str(desc='the output file/folder')


class ReconDerivativesDataSink(DerivativesDataSink):
    input_spec = _ReconDerivativesDataSinkInputSpec
    output_spec = _ReconDerivativesDataSinkOutputSpec
    out_path_base = 'qsirecon'

    def _run_interface(self, runtime):

        # If there is no qsirecon suffix, then we're not saving this file
        if not self.inputs.qsirecon_suffix:
            return runtime

        # Figure out what the extension should be based on the input file and compression
        source_file = self.inputs.source_file
        if not isinstance(source_file, str):
            source_file = source_file[0]

        src_fname, _ = _splitext(source_file)
        src_fname, dtype = src_fname.rsplit('_', 1)
        _, ext = _splitext(self.inputs.in_file[0])
        if self.inputs.compress is True and not ext.endswith('.gz'):
            ext += '.gz'
        elif self.inputs.compress is False and ext.endswith('.gz'):
            ext = ext[:-3]

        # Prepare the bids entities from the inputs
        output_bids = {}
        if self.inputs.atlas:
            output_bids['atlas'] = self.inputs.atlas
        if self.inputs.space:
            output_bids['space'] = self.inputs.space
        if self.inputs.bundles:
            output_bids['bundles'] = self.inputs.bundles
        if self.inputs.bundle:
            output_bids['bundle'] = self.inputs.bundle
        if self.inputs.space:
            output_bids['space'] = self.inputs.space
        if self.inputs.model:
            output_bids['model'] = self.inputs.model
        if self.inputs.param:
            output_bids['param'] = self.inputs.param
        if self.inputs.suffix:
            output_bids['suffix'] = self.inputs.suffix
        if self.inputs.label:
            output_bids['label'] = self.inputs.label
        if self.inputs.extension:
            output_bids['extension'] = self.inputs.extension

        # Get the output name without an extension
        bname = get_recon_output_name(
            base_dir=self.inputs.base_directory,
            source_file=source_file,
            derivative_file=self.inputs.in_file[0],
            output_bids_entities=output_bids,
            use_ext=False,
            dismiss_entities=self.inputs.dismiss_entities,
        )

        # Ensure the directory exists
        os.makedirs(op.dirname(bname), exist_ok=True)

        formatstr = '{bname}{ext}'
        # If the derivative is a directory, copy it over
        copy_dir = op.isdir(str(self.inputs.in_file[0]))
        if copy_dir:
            out_file = formatstr.format(bname=bname, ext='')
            copytree(str(self.inputs.in_file), out_file, dirs_exist_ok=True)
            self._results['out_file'] = out_file
            return runtime

        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = '{bname}{i:04d}{ext}'

        # Otherwise it's file(s)
        self._results['compression'] = []
        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(bname=bname, i=i, ext=ext)
            if isdefined(self.inputs.extra_values):
                out_file = out_file.format(extra_value=self.inputs.extra_values[i])
            self._results['out_file'].append(out_file)
            self._results['compression'].append(_copy_any(fname, out_file))
        return runtime


def _splitext(fname):
    fname, ext = op.splitext(op.basename(fname))
    if ext == '.gz':
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext
    return fname, ext


def _copy_any(src, dst):
    from nipype.utils.filemanip import copyfile

    src_isgz = src.endswith('.gz')
    dst_isgz = dst.endswith('.gz')
    if src_isgz == dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    dst_open = gzip.open if dst_isgz else open
    with src_open(src, 'rb') as f_in:
        with dst_open(dst, 'wb') as f_out:
            copyfileobj(f_in, f_out)
    return True


def get_entity(filename, entity):
    """Extract a given entity from a BIDS filename via string manipulation.

    Parameters
    ----------
    filename : :obj:`str`
        Path to the BIDS file.
    entity : :obj:`str`
        The entity to extract from the filename.

    Returns
    -------
    entity_value : :obj:`str` or None
        The BOLD file's entity value associated with the requested entity.
    """
    import os
    import re

    folder, file_base = os.path.split(filename)

    # Allow + sign, which is not allowed in BIDS,
    # but is used by templateflow for the MNIInfant template.
    entity_values = re.findall(f'{entity}-([a-zA-Z0-9+]+)', file_base)
    entity_value = None if len(entity_values) < 1 else entity_values[0]
    if entity == 'space' and entity_value is None:
        foldername = os.path.basename(folder)
        if foldername == 'anat':
            entity_value = 'T1w'
        elif foldername == 'func':
            entity_value = 'native'
        else:
            raise ValueError(f'Unknown space for {filename}')

    return entity_value
