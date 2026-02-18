# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copied recent function write_bidsignore
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#

import json
import os
import sys
import warnings
from pathlib import Path

import filelock
from bids import BIDSLayout
from bids.utils import listify
from nipype.interfaces.base import isdefined
from nipype.interfaces.utility.base import _ravel
from nipype.pipeline import engine as pe
from packaging.version import Version

from .. import config

LOGGER = config.loggers.utils


class BIDSError(ValueError):
    def __init__(self, message, bids_root):
        indent = 10
        header = '{sep} BIDS root folder: "{bids_root}" {sep}'.format(
            bids_root=bids_root, sep=''.join(['-'] * indent)
        )
        self.msg = '\n{header}\n{indent}{message}\n{footer}'.format(
            header=header,
            indent=''.join([' '] * (indent + 1)),
            message=message,
            footer=''.join(['-'] * len(header)),
        )
        super().__init__(self.msg)
        self.bids_root = bids_root


class BIDSWarning(RuntimeWarning):
    pass


def collect_participants(bids_dir, participant_label=None, strict=False, bids_validate=True):
    """
    List the participants under the BIDS root and checks that participants
    designated with the participant_label argument exist in that folder.

    Returns the list of participants to be finally processed.

    Requesting all subjects in a BIDS directory root:

    >>> collect_participants('ds114')
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    Requesting two subjects, given their IDs:

    >>> collect_participants('ds114', participant_label=['02', '04'])
    ['02', '04']

    Requesting two subjects, given their IDs (works with 'sub-' prefixes):

    >>> collect_participants('ds114', participant_label=['sub-02', 'sub-04'])
    ['02', '04']

    Requesting two subjects, but one does not exist:

    >>> collect_participants('ds114', participant_label=['02', '14'])
    ['02']

    >>> collect_participants('ds114', participant_label=['02', '14'],
    ...                      strict=True)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    qsirecon.utils.bids.BIDSError:
    ...


    """
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        raise Exception('A layout is required')

    all_participants = set(layout.get_subjects())

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            'Could not find participants. Please make sure the BIDS data '
            'structure is present and correct. Datasets can be validated '
            'online using the BIDS Validator '
            '(http://incf.github.io/bids-validator/).\n'
            'If you are using Docker for Mac or Docker for Windows, you '
            'may need to adjust your "File sharing" preferences.',
            bids_dir,
        )

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith('sub-') else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & set(all_participants))
    if not found_label:
        raise BIDSError(
            'Could not find participants [{}]'.format(', '.join(participant_label)), bids_dir
        )

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - set(all_participants))
    if notfound_label:
        exc = BIDSError(
            'Some participants were not found: {}'.format(', '.join(notfound_label)), bids_dir
        )
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label


def collect_anatomical_data(
    *,
    layout,
    subject_id,
    session_id,
    needs_t1w_transform,
    infant_mode=False,
    bids_filters=None,
):
    """Gather any high-res anatomical data (images, transforms, segmentations) to use
    in recon workflows.

    This function searches through input data to see what anatomical data is available.
    The anatomical data may be in a freesurfer directory.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        BIDSLayout object
    subject_id : :obj:`str`
        Single subject label
    needs_t1w_transform : :obj:`bool`
        Whether a T1w transform is needed
    bids_filters : :obj:`dict`, optional
        BIDS query filters
    infant_mode : :obj:`bool`, optional
        Whether the subject is an infant

    Returns
    -------
    anat_data : :obj:`dict`
        Dictionary of anatomical data
    status : :obj:`dict`
        Dictionary of status flags

    Notes
    -----
    We'll probably want to allow/collect multiple output space
    transforms in the future.
    """
    import yaml

    from qsirecon.data import load as load_data

    anat_data = {}
    status = {'template_output_space': None}

    if session_id is None:
        LOGGER.warning('Assuming no anatomical data.')
        return anat_data, {'has_qsiprep_t1w': False, 'has_qsiprep_t1w_transforms': False}

    _spec = yaml.safe_load(load_data.readable('io_spec.yaml').read_text())
    queries = _spec['queries']['anat']

    if infant_mode:
        queries['acpc_to_template_xfm']['to'] = 'MNIInfant'
        queries['template_to_acpc_xfm']['from'] = 'MNIInfant'

    # Apply filters. These may override anything.
    bids_filters = bids_filters or {}
    for acq in queries.keys():
        if acq in bids_filters:
            queries[acq].update(bids_filters[acq])

    for name, query in queries.items():
        files = layout.get(
            return_type='file',
            subject=subject_id,
            session=session_id,
            **query,
        )
        if len(files) == 1:
            anat_data[name] = files[0]
        elif len(files) > 1:
            files_str = '\n\t'.join(files)
            raise ValueError(
                f'More than one {name} found.\nFiles found:\n\t{files_str}\nQuery: {query}'
            )
        else:
            anat_data[name] = None

    # Identify the found anatomical files.
    LOGGER.info(
        (
            f'Collected anatomical data:\n'
            f'{yaml.dump(anat_data, default_flow_style=False, indent=4)}'
        ),
    )

    status['has_qsiprep_t1w'] = True
    if not anat_data['acpc_preproc'] or not anat_data['acpc_brain_mask']:
        LOGGER.warning('No preprocessed anatomical image or brain mask found.')
        status['has_qsiprep_t1w'] = False

    status['has_qsiprep_t1w_transforms'] = True
    if not anat_data['acpc_to_template_xfm'] or not anat_data['template_to_acpc_xfm']:
        if needs_t1w_transform:
            raise ValueError(
                'Reconstruction workflow requires a T1w/ACPC-to-template transform. '
                'None were found.'
            )

        LOGGER.warning('No anat-to-template or template-to-anat transforms found.')
        status['has_qsiprep_t1w_transforms'] = False
    else:
        # Determine the output space from the transform file
        status['template_output_space'] = _determine_output_space(anat_data)
    return anat_data, status


def _determine_output_space(status):
    """Determine what output space the transform maps to/from"""
    if not status['template_to_acpc_xfm']:
        return None

    return get_entity(status['template_to_acpc_xfm'], 'from')


def write_derivative_description(
    bids_dir,
    deriv_dir,
    dataset_links=None,
):
    """Write dataset_description.json file for derivatives.

    Parameters
    ----------
    bids_dir : :obj:`str`
        Path to the BIDS derivative dataset being processed.
    deriv_dir : :obj:`str`
        Path to the output QSIRecon dataset.
    dataset_links : :obj:`dict`, optional
        Dictionary of dataset links to include in the dataset description.
    """
    from qsirecon import __version__

    # Keys deriving from source dataset
    orig_dset_description = os.path.join(bids_dir, 'dataset_description.json')
    if os.path.isfile(orig_dset_description):
        with open(orig_dset_description) as fobj:
            desc = json.load(fobj)
    else:
        LOGGER.warning(f'Dataset description DNE: {orig_dset_description}')
        desc = {}

    # Check if the dataset type is derivative
    if 'DatasetType' not in desc.keys():
        LOGGER.warning(f"DatasetType key not in {orig_dset_description}. Assuming 'derivative'.")
        desc['DatasetType'] = 'derivative'

    if desc.get('DatasetType', 'derivative') != 'derivative':
        raise ValueError(
            f"DatasetType key in {orig_dset_description} is not 'derivative'. "
            'QSIRecon only works on derivative datasets.'
        )

    # Update dataset description
    desc['Name'] = 'QSIRecon output'
    DOWNLOAD_URL = f'https://github.com/PennLINC/qsirecon/archive/{__version__}.tar.gz'
    generated_by = desc.get('GeneratedBy', [])
    generated_by.insert(
        0,
        {
            'Name': 'qsirecon',
            'Version': __version__,
            'CodeURL': DOWNLOAD_URL,
        },
    )
    desc['GeneratedBy'] = generated_by
    desc['HowToAcknowledge'] = 'Include the generated boilerplate in the methods section.'

    # Keys that can only be set by environment
    if 'QSIRECON_DOCKER_TAG' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'docker',
            'Tag': f'pennlinc/qsirecon:{os.environ["QSIRECON_DOCKER_TAG"]}',
        }
    elif 'QSIRECON_SINGULARITY_URL' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'singularity',
            'URI': os.getenv('QSIRECON_SINGULARITY_URL'),
        }

    if 'DatasetDOI' in desc:
        desc['SourceDatasetsURLs'] = [f'https://doi.org/{desc["DatasetDOI"]}']

    dataset_links = dataset_links.copy()

    # Replace local templateflow path with URL
    dataset_links['templateflow'] = 'https://github.com/templateflow/templateflow'

    # Don't inherit DatasetLinks from preprocessing derivatives
    desc['DatasetLinks'] = {k: str(v) for k, v in dataset_links.items()}

    out_dset_description = os.path.join(deriv_dir, 'dataset_description.json')
    lock_file = os.path.join(deriv_dir, 'qsirecon_dataset_description.lock')
    with filelock.SoftFileLock(lock_file, timeout=60):
        if os.path.isfile(out_dset_description):
            with open(out_dset_description) as fo:
                old_dset_desc = json.load(fo)

            old_version = old_dset_desc['GeneratedBy'][0]['Version']
            if Version(__version__).public != Version(old_version).public:
                LOGGER.warning(f'Previous output generated by version {old_version} found.')
        else:
            with open(out_dset_description, 'w') as fo:
                json.dump(desc, fo, indent=4, sort_keys=True)


def write_atlas_dataset_description(atlas_dir):
    """Write dataset_description.json file for Atlas derivatives.

    Parameters
    ----------
    atlas_dir : :obj:`str`
        Path to the output QSIRecon Atlases dataset.
    """
    import json
    import os

    from qsirecon import __version__

    DOWNLOAD_URL = f'https://github.com/PennLINC/qsirecon/archive/{__version__}.tar.gz'

    desc = {
        'Name': 'QSIRecon Atlases',
        'DatasetType': 'atlas',
        'GeneratedBy': [
            {
                'Name': 'qsirecon',
                'Version': __version__,
                'CodeURL': DOWNLOAD_URL,
            },
        ],
        'HowToAcknowledge': 'Include the generated boilerplate in the methods section.',
    }
    os.makedirs(atlas_dir, exist_ok=True)

    atlas_dset_description = os.path.join(atlas_dir, 'dataset_description.json')
    if os.path.isfile(atlas_dset_description):
        with open(atlas_dset_description) as fo:
            old_desc = json.load(fo)

        old_version = old_desc['GeneratedBy'][0]['Version']
        if Version(__version__).public != Version(old_version).public:
            LOGGER.warning(f'Previous output generated by version {old_version} found.')

    else:
        with open(atlas_dset_description, 'w') as fo:
            json.dump(desc, fo, indent=4, sort_keys=True)


def write_bidsignore(deriv_dir):
    bids_ignore = (
        '*.html',
        'logs/',
        'figures/',  # Reports
        '*_xfm.*',  # Unspecified transform files
        '*.surf.gii',  # Unspecified structural outputs
        # Unspecified functional outputs
        '*_boldref.nii.gz',
        '*_bold.func.gii',
        '*_mixing.tsv',
        '*_timeseries.tsv',
    )
    ignore_file = Path(deriv_dir) / '.bidsignore'

    lock_file = os.path.join(deriv_dir, 'qsirecon_bidsignore.lock')
    with filelock.SoftFileLock(lock_file, timeout=60):
        ignore_file.write_text('\n'.join(bids_ignore) + '\n')


def validate_input_dir(exec_env, bids_dir, participant_label):
    # Ignore issues and warnings that should not influence qsirecon
    import subprocess
    import tempfile

    validator_config_dict = {
        'ignore': [
            'EVENTS_COLUMN_ONSET',
            'EVENTS_COLUMN_DURATION',
            'TSV_EQUAL_ROWS',
            'TSV_EMPTY_CELL',
            'TSV_IMPROPER_NA',
            'VOLUME_COUNT_MISMATCH',
            'INCONSISTENT_SUBJECTS',
            'INCONSISTENT_PARAMETERS',
            'PARTICIPANT_ID_COLUMN',
            'PARTICIPANT_ID_MISMATCH',
            'TASK_NAME_MUST_DEFINE',
            'PHENOTYPE_SUBJECTS_MISSING',
            'STIMULUS_FILE_MISSING',
            'EVENTS_TSV_MISSING',
            'TSV_IMPROPER_NA',
            'ACQTIME_FMT',
            'Participants age 89 or higher',
            'DATASET_DESCRIPTION_JSON_MISSING',
            'FILENAME_COLUMN',
            'WRONG_NEW_LINE',
            'MISSING_TSV_COLUMN_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_ELECTRODES',
            'UNUSED_STIMULUS',
            'CHANNELS_COLUMN_SFREQ',
            'CHANNELS_COLUMN_LOWCUT',
            'CHANNELS_COLUMN_HIGHCUT',
            'CHANNELS_COLUMN_NOTCH',
            'CUSTOM_COLUMN_WITHOUT_DESCRIPTION',
            'ACQTIME_FMT',
            'SUSPICIOUSLY_LONG_EVENT_DESIGN',
            'SUSPICIOUSLY_SHORT_EVENT_DESIGN',
            'MISSING_TSV_COLUMN_EEG_ELECTRODES',
            'MISSING_SESSION',
            'NO_T1W',
        ],
        'ignoredFiles': ['/README', '/dataset_description.json', '/participants.tsv'],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = {s.name[4:] for s in bids_dir.glob('sub-*')}
        selected_subs = {s[4:] if s.startswith('sub-') else s for s in participant_label}
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                'Data for requested participant(s) label(s) not found. Could '
                'not find data for participant(s): %s. Please verify the requested '
                'participant labels.'
            )
            if exec_env == 'docker':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the docker container. Please make sure all '
                    'volumes are mounted properly (see https://docs.docker.com/'
                    'engine/reference/commandline/run/#mount-volume--v---read-only)'
                )
            if exec_env == 'singularity':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the singularity container. Please make sure '
                    'all paths are mapped properly (see https://www.sylabs.io/'
                    'guides/3.0/user-guide/bind_paths_and_mounts.html)'
                )
            raise RuntimeError(error_msg % ','.join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict['ignoredFiles'].append(f'/sub-{sub}/**')
    with tempfile.NamedTemporaryFile('w+') as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(['bids-validator', bids_dir, '-c', temp.name])
        except FileNotFoundError:
            print('bids-validator does not appear to be installed', file=sys.stderr)


def _get_shub_version(singularity_url):
    raise ValueError('Not yet implemented')


def clean_datasinks(workflow: pe.Workflow, qsirecon_suffix: str | None) -> pe.Workflow:
    """Overwrite the base_directory of Datasinks."""
    out_dir = Path(config.execution.output_dir)
    if qsirecon_suffix:
        out_dir = out_dir / 'derivatives' / f'qsirecon-{qsirecon_suffix}'

    for node in workflow.list_node_names():
        node_name = node.split('.')[-1]
        if node_name.startswith('ds_') or node_name.startswith('_ds_'):
            workflow.get_node(node).inputs.base_directory = str(out_dir)

    return workflow


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


def get_iterable_dwis_and_anats(layout):
    """Look through the BIDS Layout for DWIs and their corresponding anats.

    Returns
    -------
    dwis_and_anats : list of tuple
        List of two-element tuples where the first element is a DWI scan and the second is
        the corresponding anatomical scan.
    """
    from bids.layout import Query

    dwis_and_anats = []
    dwi_files = layout.get(
        suffix='dwi',
        session=Query.OPTIONAL,
        space=['T1w', 'ACPC'],
        extension=['nii', 'nii.gz'],
    )

    for dwi_scan in dwi_files:
        subject_level_anats = layout.get(
            suffix=['T1w', 'T2w'],
            session=Query.NONE,
            space=[Query.NONE, 'ACPC'],
            extension=['nii', 'nii.gz'],
        )

        session_level_anats = []
        if dwi_session := dwi_scan.entities.get('session'):
            session_level_anats = layout.get(
                suffix=['T1w', 'T2w'],
                session=dwi_session,
                space=[Query.NONE, 'ACPC'],
                extension=['nii', 'nii.gz'],
            )

        if not (session_level_anats or subject_level_anats):
            anat_scan = None
        else:
            best_anat_source = session_level_anats if session_level_anats else subject_level_anats
            anat_scan = best_anat_source[0]

        dwis_and_anats.append((dwi_scan, anat_scan))
    return dwis_and_anats


def _find_nearest_path(path_dict, input_path):
    """Find the nearest relative path from an input path to a dictionary of paths.

    If ``input_path`` is not relative to any of the paths in ``path_dict``,
    the absolute path string is returned.
    If ``input_path`` is already a BIDS-URI, then it will be returned unmodified.

    Parameters
    ----------
    path_dict : dict of (str, Path)
        A dictionary of paths.
    input_path : Path
        The input path to match.

    Returns
    -------
    matching_path : str
        The nearest relative path from the input path to a path in the dictionary.
        This is either the concatenation of the associated key from ``path_dict``
        and the relative path from the associated value from ``path_dict`` to ``input_path``,
        or the absolute path to ``input_path`` if no matching path is found from ``path_dict``.

    Examples
    --------
    >>> from pathlib import Path
    >>> path_dict = {
    ...     'bids::': Path('/data/derivatives/fmriprep'),
    ...     'bids:raw:': Path('/data'),
    ...     'bids:deriv-0:': Path('/data/derivatives/source-1'),
    ... }
    >>> input_path = Path('/data/derivatives/source-1/sub-01/func/sub-01_task-rest_bold.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # match to 'bids:deriv-0:'
    'bids:deriv-0:sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> input_path = Path('/out/sub-01/func/sub-01_task-rest_bold.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # no match- absolute path
    '/out/sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> input_path = Path('/data/sub-01/func/sub-01_task-rest_bold.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # match to 'bids:raw:'
    'bids:raw:sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> input_path = 'bids::sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> _find_nearest_path(path_dict, input_path)  # already a BIDS-URI
    'bids::sub-01/func/sub-01_task-rest_bold.nii.gz'
    """
    # Don't modify BIDS-URIs
    if isinstance(input_path, str) and input_path.startswith('bids:'):
        return input_path

    input_path = Path(input_path)
    matching_path = None
    for key, path in path_dict.items():
        if input_path.is_relative_to(path):
            relative_path = input_path.relative_to(path)
            if (matching_path is None) or (len(relative_path.parts) < len(matching_path.parts)):
                matching_key = key
                matching_path = relative_path

    if matching_path is None:
        matching_path = str(input_path.absolute())
    else:
        matching_path = f'{matching_key}{matching_path}'

    return matching_path


def _get_bidsuris(in_files, dataset_links, out_dir):
    """Convert input paths to BIDS-URIs using a dictionary of dataset links."""
    in_files = listify(in_files)
    in_files = _ravel(in_files)
    # Remove undefined inputs
    in_files = [f for f in in_files if isdefined(f)]
    # Convert the dataset links to BIDS URI prefixes
    updated_keys = {f'bids:{k}:': Path(v) for k, v in dataset_links.items()}
    updated_keys['bids::'] = Path(out_dir)
    # Convert the paths to BIDS URIs
    out = [_find_nearest_path(updated_keys, f) for f in in_files]
    return out
