# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Changes made to parse QSIRecon cli arguments
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
"""Parser."""

import os
from argparse import Action
from pathlib import Path

from .. import config


class ToDict(Action):
    """A custom argparse "store" action to handle a list of key=value pairs."""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: U100
        """Call the argument."""
        d = {}
        for spec in values:
            try:
                name, loc = spec.split("=")
                loc = Path(loc)
            except ValueError:
                loc = Path(spec)
                name = loc.name

            if name in d:
                raise parser.error(f"Received duplicate derivative name: {name}")
            elif name == "preprocessed":
                raise parser.error("The 'preprocessed' derivative is reserved for internal use.")

            d[name] = loc
        setattr(namespace, self.dest, d)


def _build_parser(**kwargs):
    """Build parser object.

    ``kwargs`` are passed to ``argparse.ArgumentParser`` (mainly useful for debugging).
    """
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from functools import partial
    from pathlib import Path

    from packaging.version import Version

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>.")
        return Path(path).absolute()

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = _path_exists(path, parser)
        if not path.is_file():
            raise parser.error(f"Path should point to a file (or symlink of file): <{path}>.")
        return path

    def _min_one(value, parser):
        """Ensure an argument is not lower than 1."""
        value = int(value)
        if value < 1:
            raise parser.error("Argument can't be less than one.")
        return value

    def _to_gb(value):
        scale = {"G": 1, "T": 10**3, "M": 1e-3, "K": 1e-6, "B": 1e-9}
        digits = "".join([c for c in value if c.isdigit()])
        units = value[len(digits) :] or "M"
        return int(digits) * scale[units[0]]

    def _drop_sub(value):
        return value[4:] if value.startswith("sub-") else value

    def _drop_ses(value):
        return value[4:] if value.startswith("ses-") else value

    def _process_value(value):
        import bids

        if value is None:
            return bids.layout.Query.NONE
        elif value == "*":
            return bids.layout.Query.ANY
        else:
            return value

    def _filter_pybids_none_any(dct):
        d = {}
        for k, v in dct.items():
            if isinstance(v, list):
                d[k] = [_process_value(val) for val in v]
            else:
                d[k] = _process_value(v)
        return d

    def _bids_filter(value, parser):
        from json import JSONDecodeError, loads

        if value:
            if Path(value).exists():
                try:
                    return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
                except JSONDecodeError as e:
                    raise parser.error(f"JSON syntax error in: <{value}>.") from e
            else:
                raise parser.error(f"Path does not exist: <{value}>.")

    verstr = f"QSIRecon v{config.environment.version}"
    currentv = Version(config.environment.version)
    is_release = not any((currentv.is_devrelease, currentv.is_prerelease, currentv.is_postrelease))

    parser = ArgumentParser(
        description=f"{verstr}: q-Space Image Reconstruction Workflows",
        formatter_class=ArgumentDefaultsHelpFormatter,
        **kwargs,
    )
    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)
    BIDSFilter = partial(_bids_filter, parser=parser)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        "input_dir",
        action="store",
        metavar="input_dir",
        type=PathExists,
        help=(
            "The root folder of the input dataset  "
            "(subject-level folders should be found at the top level in this folder). "
            "If the dataset is not BIDS-valid, "
            "then a BIDS-compliant version will be created based on the --input-type value."
        ),
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The output path for the outcomes of postprocessing and visual reports",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant"],
        help='Processing stage to be run, only "participant" in the case of QSIRecon (for now).',
    )

    g_bids = parser.add_argument_group("Options for filtering input data")
    g_bids.add_argument(
        "--participant-label",
        action="store",
        nargs="+",
        type=_drop_sub,
        help="A space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    g_bids.add_argument(
        "--session-id",
        action="store",
        nargs="+",
        type=_drop_ses,
        default=None,
        help="A space delimited list of session identifiers or a single "
        "identifier (the ses- prefix can be removed)",
    )

    g_bids.add_argument(
        "-d",
        "--datasets",
        action=ToDict,
        metavar="PACKAGE=PATH",
        type=str,
        nargs="+",
        help=(
            "Search PATH(s) for derivatives or atlas datasets. "
            "These may be provided as named folders "
            "(e.g., ``--datasets smriprep=/path/to/smriprep``)."
        ),
    )
    g_bids.add_argument(
        "--bids-filter-file",
        dest="bids_filters",
        action="store",
        type=BIDSFilter,
        metavar="FILE",
        help="A JSON file describing custom BIDS input filters using PyBIDS. "
        "For further details, please check out "
        "https://fmriprep.readthedocs.io/en/%s/faq.html#"
        "how-do-I-select-only-certain-files-to-be-input-to-fMRIPrep"
        % (currentv.base_version if is_release else "latest"),
    )
    g_bids.add_argument(
        "--bids-database-dir",
        metavar="PATH",
        type=Path,
        help="Path to a PyBIDS database folder, for faster indexing (especially "
        "useful for large datasets). Will be created if not present.",
    )

    g_perfm = parser.add_argument_group("Options to handle performance")
    g_perfm.add_argument(
        "--nprocs",
        "--nthreads",
        "--n-cpus",
        dest="nprocs",
        action="store",
        type=PositiveInt,
        help="Maximum number of threads across all processes",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        action="store",
        type=PositiveInt,
        help="Maximum number of threads per-process",
    )
    g_perfm.add_argument(
        "--mem",
        "--mem-mb",
        dest="memory_gb",
        action="store",
        type=_to_gb,
        metavar="MEMORY_MB",
        help="Upper bound memory limit for QSIRecon processes",
    )
    g_perfm.add_argument(
        "--low-mem",
        action="store_true",
        help="Attempt to reduce memory usage (will increase disk usage in working directory)",
    )
    g_perfm.add_argument(
        "--use-plugin",
        "--nipype-plugin-file",
        action="store",
        metavar="FILE",
        type=IsFile,
        help="Nipype plugin configuration file",
    )
    g_perfm.add_argument(
        "--sloppy",
        action="store_true",
        default=False,
        help="Use low-quality tools for speed - TESTING ONLY",
    )

    g_subset = parser.add_argument_group("Options for performing only a subset of the workflow")
    g_subset.add_argument(
        "--boilerplate-only",
        "--boilerplate",
        action="store_true",
        default=False,
        help="Generate boilerplate only",
    )
    g_subset.add_argument(
        "--reports-only",
        action="store_true",
        default=False,
        help="Only generate reports, don't run workflows. This will only rerun report "
        "aggregation, not reportlet generation for specific nodes.",
    )
    g_subset.add_argument(
        "--report-output-level",
        action="store",
        choices=["root", "subject", "session"],
        default="root",
        help="Where should the html reports be written? By default root will write "
        "them to the --output-dir. Other options will write them into their "
        "subject or session directory.",
    )

    g_conf = parser.add_argument_group("Workflow configuration")
    g_conf.add_argument(
        "--infant", action="store_true", help="configure pipelines to process infant brains"
    )
    g_conf.add_argument(
        "--b0-threshold",
        action="store",
        type=int,
        default=100,
        help="any value in the .bval file less than this will be considered "
        "a b=0 image. Current default threshold = 100; this threshold can be "
        "lowered or increased. Note, setting this too high can result in inaccurate results.",
    )
    g_conf.add_argument(
        "--output-resolution",
        action="store",
        # required when not recon-only (which can be specified in sysargs 2 ways)
        required=False,
        type=float,
        help="the isotropic voxel size in mm the data will be resampled to "
        "after preprocessing. If set to a lower value than the original voxel "
        "size, your data will be upsampled using BSpline interpolation.",
    )

    # FreeSurfer options
    g_fs = parser.add_argument_group("Specific options for FreeSurfer preprocessing")
    g_fs.add_argument(
        "--fs-license-file",
        metavar="PATH",
        type=Path,
        help="Path to FreeSurfer license key file. Get it (for free) by registering "
        "at https://surfer.nmr.mgh.harvard.edu/registration.html",
    )

    # arguments for reconstructing QSI data
    g_recon = parser.add_argument_group("Options for recon workflows")
    g_recon.add_argument(
        "--recon-spec",
        action="store",
        type=str,
        help="json file specifying a reconstruction pipeline to be run after preprocessing",
    )
    g_recon.add_argument(
        "--input-type",
        action="store",
        default="qsiprep",
        choices=["qsiprep", "ukb", "hcpya"],
        help=(
            "Specify which pipeline was used to create the data specified as the input_dir."
            "Not necessary to specify if the data was processed by QSIPrep. "
            "Other options include "
            '"ukb" for data processed with the UK BioBank minimal preprocessing pipeline and '
            '"hcpya" for the HCP young adult minimal preprocessing pipeline.'
        ),
    )
    g_recon.add_argument(
        "--fs-subjects-dir",
        action="store",
        metavar="PATH",
        type=PathExists,
        help=(
            "Directory containing Freesurfer outputs to be integrated into recon. "
            "Freesurfer must already be run. QSIRecon will not run Freesurfer."
        ),
    )
    g_recon.add_argument(
        "--skip-odf-reports",
        action="store_true",
        default=False,
        help="run only reconstruction, assumes preprocessing has already completed.",
    )

    g_parcellation = parser.add_argument_group("Parcellation options")
    g_parcellation.add_argument(
        "--atlases",
        action="store",
        nargs="+",
        metavar="ATLAS",
        default=None,
        dest="atlases",
        help=(
            "Selection of atlases to apply to the data. "
            "Built-in atlases include: AAL116, AICHA384Ext, Brainnetome246Ext, "
            "Gordon333Ext, and the 4S atlases."
        ),
    )

    g_other = parser.add_argument_group("Other options")
    g_other.add_argument("--version", action="version", version=verstr)
    g_other.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increases log verbosity for each occurrence, debug level is -vvv",
    )
    g_other.add_argument(
        "-w",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("work").absolute(),
        help="Path where intermediate results should be stored",
    )
    g_other.add_argument(
        "--resource-monitor",
        action="store_true",
        default=False,
        help="Enable Nipype's resource monitoring to keep track of memory and CPU usage",
    )
    g_other.add_argument(
        "--config-file",
        action="store",
        metavar="FILE",
        help="Use pre-generated configuration file. Values in file will be overridden "
        "by command-line arguments.",
    )
    g_other.add_argument(
        "--write-graph",
        action="store_true",
        default=False,
        help="Write workflow graph.",
    )
    g_other.add_argument(
        "--stop-on-first-crash",
        action="store_true",
        default=False,
        help="Force stopping on first crash, even if a work directory was specified.",
    )
    g_other.add_argument(
        "--notrack",
        action="store_true",
        default=False,
        help=(
            "Opt-out of sending tracking information of this run to "
            "the QSIRecon developers. This information helps to "
            "improve QSIRecon and provides an indicator of real "
            "world usage crucial for obtaining funding."
        ),
    )
    g_other.add_argument(
        "--debug",
        action="store",
        nargs="+",
        choices=config.DEBUG_MODES + ("all",),
        help="Debug mode(s) to enable. 'all' is alias for all available modes.",
    )
    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    import logging

    # from niworkflows.utils.spaces import Reference, SpatialReferences

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)

    if opts.config_file:
        skip = {} if opts.reports_only else {"execution": ("run_uuid",)}
        config.load(opts.config_file, skip=skip, init=False)
        config.loggers.cli.info(f"Loaded previous configuration file {opts.config_file}")

    # Add internal atlas datasets to the list of datasets
    opts.datasets = opts.datasets or {}
    if opts.atlases:
        if "qsireconatlases" not in opts.datasets:
            opts.datasets["qsireconatlases"] = Path(
                os.getenv("QSIRECON_ATLAS", "/atlas/qsirecon_atlases")
            )

        if any(atlas.startswith("4S") for atlas in opts.atlases):
            if "qsirecon4s" not in opts.datasets:
                opts.datasets["qsirecon4s"] = Path(
                    os.getenv("QSIRECON_ATLASPACK", "/atlas/AtlasPack")
                )

    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    config.from_dict(vars(opts), init=["nipype"])

    if not config.execution.notrack:
        import importlib.util

        if importlib.util.find_spec("sentry_sdk") is None:
            config.execution.notrack = True
            config.loggers.cli.warning("Telemetry disabled because sentry_sdk is not installed.")
        else:
            config.loggers.cli.info(
                "Telemetry system to collect crashes and errors is enabled "
                "- thanks for your feedback! Use option ``--notrack`` to opt out."
            )

    # Initialize --output-spaces if not defined
    # if config.execution.output_spaces is None:
    #     config.execution.output_spaces = SpatialReferences(
    #         [Reference("MNI152NLin2009cAsym", {"res": "native"})]
    #     )

    # Retrieve logging level
    build_log = config.loggers.cli

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        import yaml

        with open(opts.use_plugin) as f:
            plugin_settings = yaml.safe_load(f)
        _plugin = plugin_settings.get("plugin")
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get("plugin_args", {})
            config.nipype.nprocs = opts.nprocs or config.nipype.plugin_args.get(
                "n_procs", config.nipype.nprocs
            )

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    if 1 < config.nipype.nprocs < config.nipype.omp_nthreads:
        build_log.warning(
            f"Per-process threads (--omp-nthreads={config.nipype.omp_nthreads}) exceed "
            f"total threads (--nthreads/--n-cpus={config.nipype.nprocs})"
        )

    input_dir = config.execution.input_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir
    version = config.environment.version

    # Ensure input and output folders are not the same
    if output_dir == input_dir:
        parser.error(
            "The selected output folder is the same as the input BIDS folder. "
            "Please modify the output path (suggestion: %s)."
            % input_dir
            / "derivatives"
            / ("qsirecon-%s" % version.split("+")[0])
        )

    if input_dir in work_dir.parents:
        parser.error(
            "The selected working directory is a subdirectory of the input BIDS folder. "
            "Please modify the output path."
        )

    # Setup directories
    log_dir = output_dir / "logs"
    # Check and create output and working directories
    log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Run ingression if necessary
    if config.workflow.input_type in ("hcpya", "ukb"):
        import shutil

        from ingress2qsirecon.data import load_resource
        from ingress2qsirecon.utils.functions import create_layout
        from ingress2qsirecon.utils.workflows import create_ingress2qsirecon_wf

        # Fake BIDS directory to be created
        config.execution.bids_dir = work_dir / "bids"

        # Make fake BIDS files
        bids_scaffold = load_resource("bids_scaffold")
        if not (config.execution.bids_dir / "dataset_description.json").exists():
            shutil.copytree(
                bids_scaffold,
                config.execution.bids_dir,
                dirs_exist_ok=True,
            )

        if config.execution.participant_label is None:
            participants_ingression = []
        else:
            participants_ingression = list(config.execution.participant_label)
        layouts = create_layout(
            config.execution.input_dir,
            config.execution.bids_dir,
            config.workflow.input_type,
            participants_ingression,
        )

        # Create the ingression workflow
        wf = create_ingress2qsirecon_wf(
            layouts,
            config.workflow.input_type,
            base_dir=work_dir,
        )

        # Configure the nipype workflow
        wf.config["execution"]["crashdump_dir"] = str(log_dir)
        wf.run()

        # Change the participants label based on ingression renaming
        if config.execution.participant_label is not None:
            config.execution.participant_label = [layout["subject"] for layout in layouts]

    else:
        config.execution.bids_dir = config.execution.input_dir

    # Update the config with an empty dict to trigger initialization of all config
    # sections (we used `init=False` above).
    # This must be done after cleaning the work directory, or we could delete an
    # open SQLite database
    config.from_dict({})
    config.execution.log_dir = log_dir

    # Force initialization of the BIDSLayout
    config.execution.init()
    all_subjects = config.execution.layout.get_subjects()
    if config.execution.participant_label is None:
        config.execution.participant_label = all_subjects

    participant_label = set(config.execution.participant_label)
    missing_subjects = participant_label - set(all_subjects)
    if missing_subjects:
        parser.error(
            "One or more participant labels were not found in the BIDS directory: "
            "%s." % ", ".join(missing_subjects)
        )

    config.execution.participant_label = sorted(participant_label)
    config.execution.processing_list = _get_iterable_dwis_and_anats()


def _get_iterable_dwis_and_anats():
    """Look through the BIDS Layout for DWIs and their corresponding anats.

    Returns
    -------
    dwis_and_anats : list of tuple
        List of two-element tuples where the first element is a DWI scan and the second is
        the corresponding anatomical scan.
    """
    from bids.layout import Query

    dwis_and_anats = []
    dwi_files = config.execution.layout.get(
        suffix="dwi",
        session=Query.OPTIONAL,
        space=["T1w", "ACPC"],
        extension=["nii", "nii.gz"],
    )

    for dwi_scan in dwi_files:
        subject_level_anats = config.execution.layout.get(
            suffix=["T1w", "T2w"],
            session=Query.NONE,
            space=[Query.NONE, "ACPC"],
            extension=["nii", "nii.gz"],
        )

        session_level_anats = []
        if dwi_session := dwi_scan.entities.get("session"):
            session_level_anats = config.execution.layout.get(
                suffix=["T1w", "T2w"],
                session=dwi_session,
                space=[Query.NONE, "ACPC"],
                extension=["nii", "nii.gz"],
            )

        if not (session_level_anats or subject_level_anats):
            anat_scan = None
            dwis_and_anats.append((dwi_scan))
        else:
            best_anat_source = session_level_anats if session_level_anats else subject_level_anats
            anat_scan = best_anat_source[0]
            dwis_and_anats.append((dwi_scan, anat_scan))
            
    return dwis_and_anats
