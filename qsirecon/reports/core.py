# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
from pathlib import Path

from bids.layout import Query
from nireports.assembler.report import Report

from .. import config, data


def run_reports(
    output_dir,
    subject_label,
    run_uuid,
    bootstrap_file=None,
    out_filename="report.html",
    reportlets_dir=None,
    errorname="report.err",
    metadata=None,
    **entities,
):
    """
    Run the reports.
    """
    robj = Report(
        output_dir,
        run_uuid,
        bootstrap_file=bootstrap_file,
        out_filename=out_filename,
        reportlets_dir=reportlets_dir,
        plugins=None,
        plugin_meta=None,
        metadata=metadata,
        **entities,
    )

    # Count nbr of subject for which report generation failed
    try:
        robj.generate_report()
    except:  # noqa: E722
        import sys
        import traceback

        # Store the list of subjects for which report generation failed
        traceback.print_exception(*sys.exc_info(), file=str(Path(output_dir) / "logs" / errorname))
        return subject_label

    return None


def generate_reports(
    output_level,
    output_dir,
    run_uuid,
    bootstrap_file=None,
    qsirecon_suffix="",
):
    """Generate reports for a list of subjects.

    Parameters
    ----------
    output_level : {'root', 'subject', 'session'}
    """

    errors = []
    bootstrap_file = data.load("reports-spec.yml") if bootstrap_file is None else bootstrap_file

    bids_filters = config.execution.bids_filters or {}
    for subject_label in config.execution.participant_label:
        subject_id = subject_label[4:] if subject_label.startswith("sub-") else subject_label

        # Extract session IDs from the processed DWIs
        sessions = config.execution.layout.get_sessions(
            subject=subject_label,
            session=config.execution.session_id,
            suffix="dwi",
            **bids_filters.get("dwi", {}),
        )
        if output_level == "session" and not sessions:
            report_dir = output_dir
            output_level = "subject"
            config.loggers.workflow.warning(
                "Session-level reports were requested, "
                "but data was found without a session level. "
                "Writing out reports to subject level."
            )
            sessions = [Query.NONE]

        for session_label in sessions:
            if session_label == Query.NONE:
                html_report = html_report = f"sub-{subject_id}.html"
                session_label = None
            else:
                html_report = html_report = f"sub-{subject_id}_ses-{session_label}.html"

            if output_level == "root":
                report_dir = output_dir
            elif output_level == "subject":
                report_dir = Path(output_dir) / f"sub-{subject_id}"
            elif output_level == "session":
                report_dir = Path(output_dir) / f"sub-{subject_id}" / f"ses-{session_label}"

            report_error = run_reports(
                report_dir,
                subject_label,
                run_uuid,
                bootstrap_file=bootstrap_file,
                out_filename=html_report,
                reportlets_dir=output_dir,
                errorname=f"report-{run_uuid}-{subject_label}.err",
                metadata={"qsirecon_suffix": qsirecon_suffix},
                subject=subject_label,
                session=session_label,
            )
            # If the report generation failed, append the subject label for which it failed
            if report_error is not None:
                errors.append(report_error)

        # Someday, when we have anatomical reports, add a section here that
        # finds sessions and makes the reports.

    return errors
