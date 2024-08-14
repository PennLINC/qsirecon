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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.mixins import reporting
from scipy.io.matlab import loadmat

from .qc import createB0_ColorFA_Mask_Sprites, createSprite4D

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
