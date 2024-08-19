#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to generate reportlets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import json
import os.path as op
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    Directory,
    File,
    InputMultiPath,
    SimpleInterface,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.mixins import reporting
from scipy.io.matlab import loadmat

from .qc import createB0_ColorFA_Mask_Sprites, createSprite4D

SUBJECT_TEMPLATE = """\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Structural images: {n_t1s:d} T1-weighted {t2w}</li>
\t\t<li>Diffusion-weighted series: inputs {n_dwis:d}, outputs {n_outputs:d}</li>
{groupings}
\t\t<li>Resampling targets: T1wACPC
\t</ul>
"""

DIFFUSION_TEMPLATE = """\t\t<h3 class="elem-title">Summary</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Phase-encoding (PE) direction: {pedir}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Coregistration Transform: {coregistration}</li>
\t\t\t<li>Denoising Method: {denoise_method}</li>
\t\t\t<li>Denoising Window: {denoise_window}</li>
\t\t\t<li>HMC Transform: {hmc_transform}</li>
\t\t\t<li>HMC Model: {hmc_model}</li>
\t\t\t<li>DWI series resampled to spaces: T1wACPC</li>
\t\t\t<li>Confounds collected: {confounds}</li>
\t\t\t<li>Impute slice threshold: {impute_slice_threshold}</li>
\t\t</ul>
{validation_reports}
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>QSIRecon version: {version}</li>
\t\t<li>QSIRecon command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""

GROUPING_TEMPLATE = """\t<ul>
\t\t<li>Output Name: {output_name}</li>
{input_files}
</ul>
"""

INTERACTIVE_TEMPLATE = """
<script src="https://unpkg.com/vue"></script>
<script src="https://nipreps.github.io/dmriprep-viewer/dmriprepReport.umd.min.js"></script>
<link rel="stylesheet" href="https://nipreps.github.io/dmriprep-viewer/dmriprepReport.css">

<div id="app">
  <demo :report="report"></demo>
</div>

<script>
var report = REPORT
  new Vue({
    components: {
      demo: dmriprepReport
    },
    data () {
      return {
        report
      }
    }
  }).$mount('#app')

</script>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc="HTML segment containing summary")


class AboutSummaryInputSpec(BaseInterfaceInputSpec):
    version = Str(desc="QSIRecon version")
    command = Str(desc="QSIRecon command")
    # Date not included - update timestamp only if version or command changes


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _generate_segment(self):
        raise NotImplementedError()

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = op.join(runtime.cwd, "report.html")
        with open(fname, "w") as fobj:
            fobj.write(segment)
        self._results["out_report"] = fname
        return runtime


class SubjectSummaryInputSpec(BaseInterfaceInputSpec):
    t1w = InputMultiPath(File(exists=True), desc="T1w structural images")
    t2w = InputMultiPath(File(exists=True), desc="T2w structural images")
    subjects_dir = Directory(desc="FreeSurfer subjects directory")
    subject_id = Str(desc="Subject ID")
    dwi_groupings = traits.Dict(desc="groupings of DWI files and their output names")
    output_spaces = traits.List(desc="Target spaces")
    template = traits.Enum("MNI152NLin2009cAsym", desc="Template space")


class SubjectSummaryOutputSpec(SummaryOutputSpec):
    # This exists to ensure that the summary is run prior to the first ReconAll
    # call, allowing a determination whether there is a pre-existing directory
    subject_id = Str(desc="FreeSurfer subject ID")


class SubjectSummary(SummaryInterface):
    input_spec = SubjectSummaryInputSpec
    output_spec = SubjectSummaryOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.subject_id):
            self._results["subject_id"] = self.inputs.subject_id
        return super(SubjectSummary, self)._run_interface(runtime)

    def _generate_segment(self):
        t2w_seg = ""
        if self.inputs.t2w:
            t2w_seg = "(+ {:d} T2-weighted)".format(len(self.inputs.t2w))

        # Add text for how the dwis are grouped
        n_dwis = 0
        n_outputs = 0
        groupings = ""
        if isdefined(self.inputs.dwi_groupings):
            for output_fname, group_info in self.inputs.dwi_groupings.items():
                n_outputs += 1
                files_desc = []
                files_desc.append(
                    "\t\t\t<li>Scan group: %s (PE Dir %s)</li><ul>"
                    % (output_fname, group_info["dwi_series_pedir"])
                )
                files_desc.append("\t\t\t\t<li>DWI Files: </li>")
                for dwi_file in group_info["dwi_series"]:
                    files_desc.append("\t\t\t\t\t<li> %s </li>" % dwi_file)
                    n_dwis += 1
                fieldmap_type = group_info["fieldmap_info"]["suffix"]
                if fieldmap_type is not None:
                    files_desc.append("\t\t\t\t<li>Fieldmap type: %s </li>" % fieldmap_type)

                    for key, value in group_info["fieldmap_info"].items():
                        files_desc.append("\t\t\t\t\t<li> %s: %s </li>" % (key, str(value)))
                        n_dwis += 1
                files_desc.append("</ul>")
                groupings += GROUPING_TEMPLATE.format(
                    output_name=output_fname, input_files="\n".join(files_desc)
                )

        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_t1s=len(self.inputs.t1w),
            t2w=t2w_seg,
            n_dwis=n_dwis,
            n_outputs=n_outputs,
            groupings=groupings,
            output_spaces="T1wACPC",
        )


class AboutSummary(SummaryInterface):
    input_spec = AboutSummaryInputSpec

    def _generate_segment(self):
        return ABOUT_TEMPLATE.format(
            version=self.inputs.version,
            command=self.inputs.command,
            date=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        )


class _InteractiveReportInputSpec(TraitedSpec):
    raw_dwi_file = File(exists=True, mandatory=True)
    processed_dwi_file = File(exists=True, mandatory=True)
    confounds_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    color_fa = File(exists=True, mandatory=True)
    carpetplot_data = File(exists=True, mandatory=True)
    series_qc_file = File(exists=True, mandatory=True)


class InteractiveReport(SimpleInterface):
    input_spec = _InteractiveReportInputSpec
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        report = {}
        report["dwi_corrected"] = createSprite4D(self.inputs.processed_dwi_file)

        b0, colorFA, mask = createB0_ColorFA_Mask_Sprites(
            self.inputs.processed_dwi_file, self.inputs.color_fa, self.inputs.mask_file
        )
        report["carpetplot"] = []
        if isdefined(self.inputs.carpetplot_data):
            with open(self.inputs.carpetplot_data, "r") as carpet_f:
                carpet_data = json.load(carpet_f)
            report.update(carpet_data)

        # Load the QC file
        report["qc_scores"] = json.loads(
            pd.read_csv(self.inputs.series_qc_file).to_json(orient="records")
        )[0]

        report["b0"] = b0
        report["colorFA"] = colorFA
        report["anat_mask"] = mask
        report["outlier_volumes"] = []
        report["eddy_params"] = [[i, i] for i in range(30)]
        eddy_qc = {}
        report["eddy_quad"] = eddy_qc
        report["subject_id"] = "sub-test"
        report["analysis_level"] = "participant"
        report["pipeline"] = "qsirecon"
        report["boilerplate"] = "boilerplate"

        df = pd.read_csv(self.inputs.confounds_file, delimiter="\t")
        translations = df[["trans_x", "trans_y", "trans_z"]].values
        rms = np.sqrt((translations**2).sum(1))
        fdisp = df["framewise_displacement"].tolist()
        fdisp[0] = None
        report["eddy_params"] = [[fd_, rms_] for fd_, rms_ in zip(fdisp, rms)]

        # Get the sampling scheme
        xyz = df[["grad_x", "grad_y", "grad_z"]].values
        bval = df["bval"].values
        qxyz = np.sqrt(bval)[:, None] * xyz
        report["q_coords"] = qxyz.tolist()
        report["color"] = _filename_to_colors(df["original_file"])

        safe_json = json.dumps(report)
        out_file = op.join(runtime.cwd, "interactive_report.json")
        with open(out_file, "w") as out_html:
            out_html.write(safe_json)
        self._results["out_report"] = out_file
        return runtime


def _filename_to_colors(labels_column, colormap="rainbow"):
    cmap = matplotlib.cm.get_cmap(colormap)
    labels, _ = pd.factorize(labels_column)
    n_samples = labels.shape[0]
    max_label = labels.max()
    if max_label == 0:
        return [(1.0, 0.0, 0.0)] * n_samples
    labels = labels / max_label
    colors = np.array([cmap(label) for label in labels])
    return colors.tolist()


class _ReconPeaksReportInputSpec(CommandLineInputSpec):
    mif_file = File(exists=True, argstr="--mif %s")
    fib_file = File(exists=True, argstr="--fib %s")
    odf_file = File(exists=True, argstr="--amplitudes %s")
    directions_file = File(exists=True, argstr="--directions %s")
    mask_file = File(exists=True, argstr="--mask_file %s")
    background_image = File(exists=True, argstr="--background_image %s")
    odf_rois = File(exists=True, argstr="--odf_rois %s")
    peak_report = File("peaks_mosaic.png", argstr="--peaks_image %s", usedefault=True)
    odf_report = File("odfs_mosaic.png", argstr="--odfs_image %s", usedefault=True)
    peaks_only = traits.Bool(
        False, usedefault=True, argstr="--peaks_only", desc="only produce a peak directions report"
    )
    subtract_iso = traits.Bool(
        False,
        usedefault=True,
        argstr="--subtract-iso",
        desc="subtract isotropic component from ODFs",
    )


class _ReconPeaksReportOutputSpec(TraitedSpec):
    peak_report = File(exists=True)
    odf_report = File()


class CLIReconPeaksReport(CommandLine):
    input_spec = _ReconPeaksReportInputSpec
    output_spec = _ReconPeaksReportOutputSpec
    _cmd = "recon_plot"
    _redirect_x = True

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["peak_report"] = op.abspath(self.inputs.peak_report)
        if not self.inputs.peaks_only:
            outputs["odf_report"] = op.abspath(self.inputs.odf_report)
        return outputs


class _ConnectivityReportInputSpec(BaseInterfaceInputSpec):
    connectivity_matfile = File(exists=True)


class _ConnectivityReportOutputSpec(reporting.ReportCapableOutputSpec):
    odf_report = File(exists=True)


class ConnectivityReport(SimpleInterface):
    input_spec = _ConnectivityReportInputSpec
    output_spec = _ConnectivityReportOutputSpec

    def _run_interface(self, runtime):
        """Generate a reportlet."""
        mat = loadmat(self.inputs.connectivity_matfile)
        connectivity_keys = [key for key in mat.keys() if key.endswith("connectivity")]
        atlases = sorted(set([key.split("_")[0] for key in connectivity_keys]))
        measures = sorted(set(["_".join(key.split("_")[1:-1]) for key in connectivity_keys]))
        nrows = len(atlases)
        ncols = len(measures)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        for connectivity_key in connectivity_keys:
            atlas = connectivity_key.split("_")[0]
            measure = "_".join(connectivity_key.split("_")[1:-1])
            row = atlases.index(atlas)
            col = measures.index(measure)
            ax[row, col].imshow(
                mat[connectivity_key], interpolation="nearest", cmap="Greys", aspect="equal"
            )
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
        fig.set_size_inches((ncols, nrows))
        fig.subplots_adjust(left=0.05, top=0.95, wspace=0, hspace=0, bottom=0, right=1)

        for measure_num, measure_name in enumerate(measures):
            ax[0, measure_num].set_title(measure_name.replace("_", "/"), fontdict={"fontsize": 6})
        for atlas_num, atlas_name in enumerate(atlases):
            ax[atlas_num, 0].set_ylabel(atlas_name, fontdict={"fontsize": 8})

        conn_report = op.join(runtime.cwd, "conn_report.svg")
        fig.savefig(conn_report)
        self._results["out_report"] = conn_report
        return runtime
