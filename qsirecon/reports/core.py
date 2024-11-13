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
    processing_list,
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
    for subject_label, session_list in processing_list.items():
        subject_id = subject_label[4:] if subject_label.startswith("sub-") else subject_label

        # Beyond a certain number of sessions per subject,
        # we separate the functional reports per session
        if session_list is None:
            all_filters = config.execution.bids_filters or {}
            filters = all_filters.get("dwi", {})
            session_list = config.execution.layout.get_sessions(subject=subject_label, **filters)

        # The number of sessions is intentionally not based on session_list but
        # on the total number of sessions, because I want the final derivatives
        # folder to be the same whether sessions were run one at a time or all-together.
        n_ses = len(session_list)
        if (n_ses > config.execution.aggr_ses_reports) or (output_level == "session"):
            session_list = [ses[4:] if ses.startswith("ses-") else ses for ses in session_list]
            for session_label in session_list:
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
                )
                # If the report generation failed, append the subject label for which it failed
                if report_error is not None:
                    errors.append(report_error)
        else:
            html_report = f"sub-{subject_id}.html"

            if output_level == "root":
                report_dir = output_dir
            elif output_level == "subject":
                report_dir = Path(output_dir) / f"sub-{subject_id}"

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
            )
            # If the report generation failed, append the subject label for which it failed
            if report_error is not None:
                errors.append(report_error)

    return errors
