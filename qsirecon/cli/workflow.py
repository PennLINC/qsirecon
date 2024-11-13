# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Changed to build qsirecon and qsirecon workflows
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
"""
The workflow builder factory method.

All the checks and the construction of the workflow are done
inside this function that has pickleable inputs and output
dictionary (``retval``) to allow isolation using a
``multiprocessing.Process`` that allows qsirecon to enforce
a hard-limited memory-scope.

"""
from pathlib import Path

from pkg_resources import resource_filename as pkgrf


def build_workflow(config_file, retval):
    """Create the Nipype Workflow that supports the whole execution graph."""

    from niworkflows.utils.bids import collect_participants

    # from niworkflows.utils.misc import check_valid_fs_license
    # from ..utils.bids import check_pipeline_version
    from .. import config
    from ..reports.core import generate_reports
    from ..utils.misc import check_deps
    from ..workflows.base import init_qsirecon_wf

    config.load(config_file)
    build_log = config.loggers.workflow

    version = config.environment.version

    retval["return_code"] = 1
    retval["workflow"] = None

    banner = [f"Running QSIRecon version {version}"]
    notice_path = Path(pkgrf("qsirecon", "data/NOTICE"))
    if notice_path.exists():
        banner[0] += "\n"
        banner += [f"License NOTICE {'#' * 50}"]
        banner += [f"QSIRecon {version}"]
        banner += notice_path.read_text().splitlines(keepends=False)[1:]
        banner += ["#" * len(banner[1])]
    build_log.log(25, f"\n{' ' * 9}".join(banner))

    # warn if older results exist: check for dataset_description.json in output folder
    # msg = check_pipeline_version("QSIRecon", version, output_dir / "dataset_description.json")
    # if msg is not None:
    #     build_log.warning(msg)

    # Please note this is the input folder's dataset_description.json
    dset_desc_path = config.execution.bids_dir / "dataset_description.json"
    if dset_desc_path.exists():
        from hashlib import sha256

        desc_content = dset_desc_path.read_bytes()
        config.execution.bids_description_hash = sha256(desc_content).hexdigest()

    # First check that bids_dir looks like a BIDS folder
    subject_list = collect_participants(
        config.execution.layout, participant_label=config.execution.participant_label
    )

    # Called with reports only
    if config.execution.reports_only:
        build_log.log(25, "Running --reports-only on participants %s", ", ".join(subject_list))
        bids_filters = config.execution.bids_filters or {}
        session_list = bids_filters.get("dwi", {}).get("session")

        processing_list = {config.execution.participant_label: session_list}
        failed_reports = generate_reports(
            processing_list=processing_list,
            output_level=config.execution.report_output_level,
            output_dir=config.execution.output_dir,
            run_uuid=config.execution.run_uuid,
            qsirecon_suffix="",
        )
        if failed_reports:
            config.loggers.cli.error(
                "Report generation was not successful for the following participants : %s.",
                ", ".join(failed_reports),
            )

        retval["return_code"] = len(failed_reports)
        return retval

    # Build main workflow
    init_msg = [
        "Building QSIRecon's workflow:",
        f"BIDS dataset path: {config.execution.bids_dir}.",
        f"Participant list: {subject_list}.",
        f"Run identifier: {config.execution.run_uuid}.",
    ]

    if config.execution.fs_subjects_dir:
        init_msg += [f"Pre-run FreeSurfer's SUBJECTS_DIR: {config.execution.fs_subjects_dir}."]

    build_log.log(25, f"\n{' ' * 11}* ".join(init_msg))

    # Check for FS license after building the workflow
    if not Path(config.execution.fs_license_file).exists():
        build_log.critical(
            """\
ERROR: a valid license file is required for FreeSurfer to run. QSIRecon looked for an existing \
license file at several paths, in this order: 1) command line argument ``--fs-license-file``; \
2) ``$FS_LICENSE`` environment variable; and 3) the ``$FREESURFER_HOME/license.txt`` path. Get it \
(for free) by registering at https://surfer.nmr.mgh.harvard.edu/registration.html"""
        )
        retval["return_code"] = 126  # 126 == Command invoked cannot execute.
        return retval

    # If qsirecon is being run on already preprocessed data:
    retval["workflow"] = init_qsirecon_wf()

    # Check workflow for missing commands
    missing = check_deps(retval["workflow"])
    if missing:
        build_log.critical(
            "Cannot run QSIRecon. Missing dependencies:%s",
            "\n\t* ".join([""] + [f"{cmd} (Interface: {iface})" for iface, cmd in missing]),
        )
        retval["return_code"] = 127  # 127 == command not found.
        return retval

    config.to_filename(config_file)
    build_log.info(
        "QSIRecon workflow graph with %d nodes built successfully.",
        len(retval["workflow"]._get_all_nodes()),
    )
    retval["return_code"] = 0
    return retval


def build_boilerplate(config_file, workflow):
    """Write boilerplate in an isolated process."""
    from .. import config

    config.load(config_file)
    logs_path = config.execution.output_dir / "logs"
    boilerplate = workflow.visit_desc()
    citation_files = {
        ext: logs_path / ("CITATION.%s" % ext) for ext in ("bib", "tex", "md", "html")
    }

    if boilerplate:
        # To please git-annex users and also to guarantee consistency
        # among different renderings of the same file, first remove any
        # existing one
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

    citation_files["md"].write_text(boilerplate)

    if citation_files["md"].exists():
        from subprocess import CalledProcessError, TimeoutExpired, check_call

        bib_text = Path(pkgrf("qsirecon", "data/boilerplate.bib")).read_text()
        citation_files["bib"].write_text(
            bib_text.replace("QSIRecon <version>", f"QSIRecon {config.environment.version}")
        )

        # Generate HTML file resolving citations
        cmd = [
            "pandoc",
            "-s",
            "--bibliography",
            str(citation_files["bib"]),
            "--filter",
            "pandoc-citeproc",
            "--metadata",
            'pagetitle="QSIRecon citation boilerplate"',
            str(citation_files["md"]),
            "-o",
            str(citation_files["html"]),
        ]

        config.loggers.cli.info("Generating an HTML version of the citation boilerplate...")
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning("Could not generate CITATION.html file:\n%s", " ".join(cmd))

        # Generate LaTex file resolving citations
        cmd = [
            "pandoc",
            "-s",
            "--bibliography",
            str(citation_files["bib"]),
            "--natbib",
            str(citation_files["md"]),
            "-o",
            str(citation_files["tex"]),
        ]
        config.loggers.cli.info("Generating a LaTeX version of the citation boilerplate...")
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning("Could not generate CITATION.tex file:\n%s", " ".join(cmd))


def copy_boilerplate(in_dir, out_dir):
    import shutil

    in_logs_path = Path(in_dir) / "logs"
    out_logs_path = Path(out_dir) / "logs"
    out_logs_path.mkdir(exist_ok=True, parents=True)
    citation_files = {
        ext: in_logs_path / ("CITATION.%s" % ext) for ext in ("bib", "tex", "md", "html")
    }
    for ext, citation_file in citation_files.items():
        if citation_file.exists():
            shutil.copy(citation_file, out_logs_path / f"CITATION.{ext}")
