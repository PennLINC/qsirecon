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
from pathlib import Path

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
    InputMultiObject,
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

SUBJECT_TEMPLATE = """\
\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Structural images: {n_t1w:d} T1-weighted {t2w}</li>
\t\t<li>Diffusion-weighted series: {n_dwi:d}</li>
\t\t<li>Standard output spaces: {std_spaces}</li>
\t\t<li>Non-standard output spaces: {nstd_spaces}</li>
\t\t<li>FreeSurfer reconstruction: {freesurfer_status}</li>
\t</ul>
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>QSIRecon version: {version}</li>
\t\t<li>QSIRecon command: <code>{command}</code></li>
\t\t<li>Date postprocessed: {date}</li>
\t</ul>
</div>
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
    subjects_dir = traits.Either(
        Directory,
        None,
        desc="FreeSurfer subjects directory",
    )
    subject_id = Str(desc="Subject ID")
    dwi = InputMultiObject(
        traits.Either(File(exists=True), traits.List(File(exists=True))),
        desc="Preprocessed DWI series",
    )
    std_spaces = traits.List(Str, desc="list of standard spaces")
    nstd_spaces = traits.List(Str, desc="list of non-standard spaces")


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
        return super()._run_interface(runtime)

    def _generate_segment(self):
        if not isdefined(self.inputs.subjects_dir):
            freesurfer_status = "Not run"
        else:
            freesurfer_status = "Pre-existing directory"

        t2w_seg = ""
        if self.inputs.t2w:
            t2w_seg = f"(+ {len(self.inputs.t2w):d} T2-weighted)"

        # Add list of tasks with number of runs
        dwi_series = self.inputs.dwi if isdefined(self.inputs.dwi) else []
        dwi_series = [s[0] if isinstance(s, list) else s for s in dwi_series]

        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_t1w=len(self.inputs.t1w),
            t2w=t2w_seg,
            n_dwi=len(dwi_series),
            std_spaces=", ".join(self.inputs.std_spaces),
            nstd_spaces=", ".join(self.inputs.nstd_spaces),
            freesurfer_status=freesurfer_status,
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


XVFB_ERROR = """

ODF/Peak Plotting did not produce the expected output:
  {}

This could be due to how QSIRecon was run as a container.

If you ran Singularity/Apptainer with --containall, please also use --writable-tmpfs

"""


class _ReconPeaksReportOutputSpec(TraitedSpec):
    peak_report = File(exists=True)
    odf_report = File()


class CLIReconPeaksReport(CommandLine):
    input_spec = _ReconPeaksReportInputSpec
    output_spec = _ReconPeaksReportOutputSpec
    _cmd = "recon_plot"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        workdir = Path(".")

        # There will always be a peak report
        peaks_file = workdir / self.inputs.peak_report
        if not peaks_file.exists():
            raise Exception(XVFB_ERROR.format(peaks_file.absolute()))
        outputs["peak_report"] = str(peaks_file.absolute())

        # If there will be no ODF report, we are done
        if self.inputs.peaks_only or not isdefined(self.inputs.odf_rois):
            return outputs

        # Find the ODF report
        odfs_file = workdir / self.inputs.odf_report
        if not odfs_file.exists():
            raise Exception(XVFB_ERROR.format(odfs_file.absolute()))
        outputs["odf_report"] = str(odfs_file.absolute())

        return outputs


class _ConnectivityReportInputSpec(BaseInterfaceInputSpec):
    connectivity_matfile = File(exists=True)


class _ConnectivityReportOutputSpec(reporting.ReportCapableOutputSpec):
    out_report = File(exists=True)


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


class _ScalarReportInputSpec(BaseInterfaceInputSpec):
    underlay = File(mandatory=True, exists=True)
    scalar_maps = traits.List(File(exists=True), mandatory=True)
    scalar_metadata = traits.List(traits.Dict, mandatory=True)
    dseg = File(mandatory=False, exists=True)
    mask_file = File(mandatory=True, exists=True)


class _ScalarReportOutputSpec(reporting.ReportCapableOutputSpec):
    out_report = File(exists=True)


class ScalarReport(SimpleInterface):
    """Plot scalar maps in a matrix of images."""

    input_spec = _ScalarReportInputSpec
    output_spec = _ScalarReportOutputSpec

    def _run_interface(self, runtime):
        """Generate a reportlet."""
        import matplotlib.pyplot as plt
        from nilearn import image
        from nireports.reportlets.utils import cuts_from_bbox

        n_scalars = len(self.inputs.scalar_maps)
        fig, axes = plt.subplots(
            nrows=n_scalars,
            ncols=3,
            figsize=(43, 6 * n_scalars),
            gridspec_kw=dict(width_ratios=[6, 36, 0.25], wspace=0),
        )

        underlay = self.inputs.underlay
        resampled_underlay = image.resample_to_img(underlay, self.inputs.scalar_maps[0])
        resampled_mask = image.resample_to_img(self.inputs.mask_file, self.inputs.scalar_maps[0])

        dseg = None
        if isdefined(self.inputs.dseg):
            dseg = image.resample_to_img(
                self.inputs.dseg,
                self.inputs.scalar_maps[0],
                interpolation="nearest",
            )

        cuts = cuts_from_bbox(resampled_underlay, cuts=6)
        z_cuts = cuts["z"]
        for i_scalar, scalar_map in enumerate(self.inputs.scalar_maps):
            scalar_name = self.inputs.scalar_metadata[i_scalar]["metadata"]["Description"]
            raise Exception(scalar_map)
            plot_scalar_map(
                underlay=resampled_underlay,
                overlay=scalar_map,
                title=scalar_name,
                z_cuts=z_cuts,
                axes=axes[i_scalar, :],
                dseg=dseg,
                mask=resampled_mask,
            )

        self._results["out_report"] = op.join(runtime.cwd, "scalar_report.svg")
        fig.savefig(self._results["out_report"])
        return runtime


def plot_scalar_map(
    underlay,
    overlay,
    mask,
    title,
    z_cuts,
    axes,
    dseg=None,
    vmin=None,
    vmax=None,
    cmap="Reds",
):
    """Plot a scalar map with a histogram of the voxel-wise values."""
    import seaborn as sns
    from matplotlib import cm
    from nilearn import image, masking, plotting

    overlay_masked = masking.unmask(masking.apply_mask(overlay, mask), mask)

    if dseg is not None:
        tissue_types = ["GM", "WM", "CSF"]
        tissue_values = [1, 2, 3]
        tissue_colors = ["#1b60a5", "#2da467", "#9d8f25"]
    else:
        tissue_types = ["Brain"]
        tissue_values = [1]
        tissue_colors = ["#1b60a5"]
        dseg = mask

    tissue_palette = dict(zip(tissue_types, tissue_colors))

    # Extract voxel-wise values for the histogram
    dfs = []
    for i_tissue_type, tissue_type in enumerate(tissue_types):
        tissue_type_val = tissue_values[i_tissue_type]
        mask_img = image.math_img(
            f"(img == {tissue_type_val}).astype(int)",
            img=dseg,
        )
        tissue_type_vals = masking.apply_mask(overlay, mask_img)
        df = pd.DataFrame(
            columns=["Data", "Tissue Type"],
            data=list(map(list, zip(*[tissue_type_vals, [tissue_type] * tissue_type_vals.size]))),
        )
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    ax0, ax1, ax2 = axes
    with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=3):
        sns.kdeplot(
            data=df,
            x="Data",
            palette=tissue_palette,
            hue="Tissue Type",
            fill=True,
            ax=ax0,
        )

    # We want the x-ticks of the histogram to match the colorbar ticks for the map image
    xticks = ax0.get_xticklabels()
    xlim = list(ax0.get_xlim())
    if vmin is not None:
        xlim[0] = vmin

    if vmax is not None:
        xlim[1] = vmax

    ax0.set_xlim(xlim)
    ax0.set_title(title)

    xticks = [
        i for i in xticks if i.get_position()[0] <= xlim[1] and i.get_position()[0] >= xlim[0]
    ]
    xticklabels = [xtick.get_text() for xtick in xticks]
    xticks = [xtick.get_position()[0] for xtick in xticks]
    xmin = xlim[0]
    xmax = xlim[1]

    # Plot the scalar map
    # The colormap is set to Reds, but we want to use the same colormap as the histogram.
    if xmin < 0:
        kwargs = {"symmetric_cbar": True}
    else:
        kwargs = {"symmetric_cbar": False, "vmin": xmin}

    plotting.plot_stat_map(
        stat_map_img=overlay_masked,
        bg_img=underlay,
        resampling_interpolation="nearest",
        display_mode="z",
        cut_coords=z_cuts,
        threshold=0.00001,
        draw_cross=False,
        colorbar=False,
        black_bg=False,
        vmax=xmax,
        axes=ax1,
        cmap=cmap,
        **kwargs,
    )
    mappable = cm.ScalarMappable(norm=plt.Normalize(vmin=xmin, vmax=xmax), cmap=cmap)
    cbar = plt.colorbar(cax=ax2, mappable=mappable)
    cbar.set_ticks(xticks)
    cbar.set_ticklabels(xticklabels)
    return
