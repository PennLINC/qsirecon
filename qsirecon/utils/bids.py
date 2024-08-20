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
from typing import Union

from bids import BIDSLayout
from nipype.pipeline import engine as pe
from packaging.version import Version

from .. import config


class BIDSError(ValueError):
    def __init__(self, message, bids_root):
        indent = 10
        header = '{sep} BIDS root folder: "{bids_root}" {sep}'.format(
            bids_root=bids_root, sep="".join(["-"] * indent)
        )
        self.msg = "\n{header}\n{indent}{message}\n{footer}".format(
            header=header,
            indent="".join([" "] * (indent + 1)),
            message=message,
            footer="".join(["-"] * len(header)),
        )
        super(BIDSError, self).__init__(self.msg)
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
        raise Exception("A layout is required")

    all_participants = set(layout.get_subjects())

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            "Could not find participants. Please make sure the BIDS data "
            "structure is present and correct. Datasets can be validated "
            "online using the BIDS Validator "
            "(http://incf.github.io/bids-validator/).\n"
            "If you are using Docker for Mac or Docker for Windows, you "
            'may need to adjust your "File sharing" preferences.',
            bids_dir,
        )

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith("sub-") else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & set(all_participants))
    if not found_label:
        raise BIDSError(
            "Could not find participants [{}]".format(", ".join(participant_label)), bids_dir
        )

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - set(all_participants))
    if notfound_label:
        exc = BIDSError(
            "Some participants were not found: {}".format(", ".join(notfound_label)), bids_dir
        )
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label


def write_derivative_description(bids_dir, deriv_dir, dataset_links=None):
    from qsirecon import __version__

    # Keys deriving from source dataset
    orig_dset_description = os.path.join(bids_dir, "dataset_description.json")
    if os.path.exists(orig_dset_description):
        with open(orig_dset_description) as fobj:
            dset_desc = json.load(fobj)
    else:
        config.loggers.utils.warning(f"Dataset description DNE: {orig_dset_description}")
        dset_desc = {}

    # Check if the dataset type is derivative
    if "DatasetType" not in dset_desc.keys():
        config.loggers.utils.warning(
            f"DatasetType key not in {orig_dset_description}. Assuming 'derivative'."
        )
        dset_desc["DatasetType"] = "derivative"

    if dset_desc.get("DatasetType", "derivative") != "derivative":
        raise ValueError(
            f"DatasetType key in {orig_dset_description} is not 'derivative'. "
            "QSIRecon only works on derivative datasets."
        )

    dset_desc["Name"] = "QSIRecon output"
    DOWNLOAD_URL = f"https://github.com/PennLINC/qsirecon/archive/{__version__}.tar.gz"
    generated_by = dset_desc.get("GeneratedBy", [])
    generated_by.insert(
        0,
        {
            "Name": "qsirecon",
            "Version": __version__,
            "CodeURL": DOWNLOAD_URL,
        },
    )
    dset_desc["GeneratedBy"] = generated_by
    dset_desc["HowToAcknowledge"] = "Include the generated boilerplate in the methods section."

    # Keys that can only be set by environment
    if "FMRIPREP_DOCKER_TAG" in os.environ:
        dset_desc["GeneratedBy"][0]["Container"] = {
            "Type": "docker",
            "Tag": f"nipreps/fmriprep:{os.environ['FMRIPREP_DOCKER_TAG']}",
        }
    elif "FMRIPREP_SINGULARITY_URL" in os.environ:
        dset_desc["GeneratedBy"][0]["Container"] = {
            "Type": "singularity",
            "URI": os.getenv("FMRIPREP_SINGULARITY_URL"),
        }

    if "DatasetDOI" in dset_desc:
        dset_desc["SourceDatasetsURLs"] = [f"https://doi.org/{dset_desc['DatasetDOI']}"]

    # Add DatasetLinks
    if "DatasetLinks" not in dset_desc.keys():
        dset_desc["DatasetLinks"] = {}

    if "preprocessed" in dset_desc["DatasetLinks"].keys():
        config.loggers.utils.warning("'preprocessed' is already a dataset link. Overwriting.")

    dset_desc["DatasetLinks"]["preprocessed"] = str(bids_dir)
    if dataset_links:
        for key, value in dataset_links.items():
            if key in dset_desc["DatasetLinks"]:
                config.loggers.utils.warning(f"'{key}' is already a dataset link. Overwriting.")

            if key == "templateflow":
                value = "https://github.com/templateflow/templateflow"

            dset_desc["DatasetLinks"][key] = str(value)

    out_dset_description = os.path.join(deriv_dir, "dataset_description.json")
    if os.path.isfile(out_dset_description):
        with open(out_dset_description, "r") as fo:
            old_dset_desc = json.load(fo)

        old_version = old_dset_desc["GeneratedBy"][0]["Version"]
        if Version(__version__).public != Version(old_version).public:
            config.loggers.utils.warning(
                f"Previous output generated by version {old_version} found."
            )

    else:
        with open(out_dset_description, "w") as fo:
            json.dump(dset_desc, fo, indent=4, sort_keys=True)


def write_bidsignore(deriv_dir):
    bids_ignore = (
        "*.html",
        "logs/",
        "figures/",  # Reports
        "*_xfm.*",  # Unspecified transform files
        "*.surf.gii",  # Unspecified structural outputs
        # Unspecified functional outputs
        "*_boldref.nii.gz",
        "*_bold.func.gii",
        "*_mixing.tsv",
        "*_timeseries.tsv",
    )
    ignore_file = Path(deriv_dir) / ".bidsignore"

    ignore_file.write_text("\n".join(bids_ignore) + "\n")


def validate_input_dir(exec_env, bids_dir, participant_label):
    # Ignore issues and warnings that should not influence qsirecon
    import subprocess
    import tempfile

    validator_config_dict = {
        "ignore": [
            "EVENTS_COLUMN_ONSET",
            "EVENTS_COLUMN_DURATION",
            "TSV_EQUAL_ROWS",
            "TSV_EMPTY_CELL",
            "TSV_IMPROPER_NA",
            "VOLUME_COUNT_MISMATCH",
            "INCONSISTENT_SUBJECTS",
            "INCONSISTENT_PARAMETERS",
            "PARTICIPANT_ID_COLUMN",
            "PARTICIPANT_ID_MISMATCH",
            "TASK_NAME_MUST_DEFINE",
            "PHENOTYPE_SUBJECTS_MISSING",
            "STIMULUS_FILE_MISSING",
            "EVENTS_TSV_MISSING",
            "TSV_IMPROPER_NA",
            "ACQTIME_FMT",
            "Participants age 89 or higher",
            "DATASET_DESCRIPTION_JSON_MISSING",
            "FILENAME_COLUMN",
            "WRONG_NEW_LINE",
            "MISSING_TSV_COLUMN_CHANNELS",
            "MISSING_TSV_COLUMN_IEEG_CHANNELS",
            "MISSING_TSV_COLUMN_IEEG_ELECTRODES",
            "UNUSED_STIMULUS",
            "CHANNELS_COLUMN_SFREQ",
            "CHANNELS_COLUMN_LOWCUT",
            "CHANNELS_COLUMN_HIGHCUT",
            "CHANNELS_COLUMN_NOTCH",
            "CUSTOM_COLUMN_WITHOUT_DESCRIPTION",
            "ACQTIME_FMT",
            "SUSPICIOUSLY_LONG_EVENT_DESIGN",
            "SUSPICIOUSLY_SHORT_EVENT_DESIGN",
            "MISSING_TSV_COLUMN_EEG_ELECTRODES",
            "MISSING_SESSION",
            "NO_T1W",
        ],
        "ignoredFiles": ["/README", "/dataset_description.json", "/participants.tsv"],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = set([s.name[4:] for s in bids_dir.glob("sub-*")])
        selected_subs = set([s[4:] if s.startswith("sub-") else s for s in participant_label])
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                "Data for requested participant(s) label(s) not found. Could "
                "not find data for participant(s): %s. Please verify the requested "
                "participant labels."
            )
            if exec_env == "docker":
                error_msg += (
                    " This error can be caused by the input data not being "
                    "accessible inside the docker container. Please make sure all "
                    "volumes are mounted properly (see https://docs.docker.com/"
                    "engine/reference/commandline/run/#mount-volume--v---read-only)"
                )
            if exec_env == "singularity":
                error_msg += (
                    " This error can be caused by the input data not being "
                    "accessible inside the singularity container. Please make sure "
                    "all paths are mapped properly (see https://www.sylabs.io/"
                    "guides/3.0/user-guide/bind_paths_and_mounts.html)"
                )
            raise RuntimeError(error_msg % ",".join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict["ignoredFiles"].append("/sub-%s/**" % sub)
    with tempfile.NamedTemporaryFile("w+") as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(["bids-validator", bids_dir, "-c", temp.name])
        except FileNotFoundError:
            print("bids-validator does not appear to be installed", file=sys.stderr)


def _get_shub_version(singularity_url):
    raise ValueError("Not yet implemented")


def clean_datasinks(workflow: pe.Workflow, qsirecon_suffix: Union[str, None]) -> pe.Workflow:
    """Overwrite the base_directory of Datasinks."""
    out_dir = Path(config.execution.output_dir)
    if qsirecon_suffix:
        out_dir = out_dir / "derivatives" / f"qsirecon-{qsirecon_suffix}"

    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).inputs.base_directory = str(out_dir)

    return workflow
