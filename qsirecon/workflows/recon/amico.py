"""
AMICO Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_amico_noddi_fit_wf

"""

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.amico import NODDI
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.converters import NODDItoFIBGZ
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import AMICOReconScalars, ReconScalarsDataSink
from ...interfaces.reports import CLIReconPeaksReport
from ...utils.bids import clean_datasinks


def init_amico_noddi_fit_wf(
    available_anatomical_data,
    name="amico_noddi_recon",
    qsirecon_suffix="",
    params={},
):
    """Reconstruct EAPs, ODFs, using 3dSHORE (brainsuite-style basis set).

    Inputs

        *qsirecon outputs*

    Outputs

        directions_image
            Image of directions
        icvf_image
            Voxelwise ICVF.
        od_image
            Voxelwise Orientation Dispersion
        isovf_image
            Voxelwise ISOVF
        config_file
            Pickle file with model configurations in it
        fibgz

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["odf_rois"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "directions_image",
                "icvf_image",
                "od_image",
                "isovf_image",
                "config_file",
                "fibgz",
                "recon_scalars",
            ],
        ),
        name="outputnode",
    )
    omp_nthreads = config.nipype.omp_nthreads
    workflow = Workflow(name=name)

    plot_reports = params.pop("plot_reports", True)
    desc = """NODDI Reconstruction

: """
    desc += """\
The NODDI model (@noddi) was fit using the AMICO implementation (@amico).
A value of %.1E was used for parallel diffusivity and %.1E for isotropic
diffusivity.""" % (
        params["dPar"],
        params["dIso"],
    )
    if params.get("is_exvivo"):
        desc += " An additional component was added to the model foe ex-vivo data."

    recon_scalars = pe.Node(
        AMICOReconScalars(qsirecon_suffix=qsirecon_suffix),
        name="recon_scalars",
        run_without_submitting=True,
    )
    noddi_fit = pe.Node(NODDI(**params), name="recon_noddi", n_procs=omp_nthreads)
    convert_to_fibgz = pe.Node(NODDItoFIBGZ(), name="convert_to_fibgz")

    workflow.connect([
        (inputnode, noddi_fit, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file'),
        ]),
        (noddi_fit, outputnode, [
            ('directions_image', 'directions_image'),
            ('icvf_image', 'icvf_image'),
            ('od_image', 'od_image'),
            ('isovf_image', 'isovf_image'),
            ('config_file', 'config_file'),
        ]),
        (noddi_fit, recon_scalars, [
            ('icvf_image', 'icvf_image'),
            ('od_image', 'od_image'),
            ('isovf_image', 'isovf_image'),
            ('directions_image', 'directions_image'),
        ]),
        (recon_scalars, outputnode, [("scalar_info", "recon_scalars")]),
        (noddi_fit, convert_to_fibgz, [
            ('directions_image', 'directions_file'),
            ('icvf_image', 'icvf_file'),
            ('od_image', 'od_file'),
            ('isovf_image', 'isovf_file'),
        ]),
        (inputnode, convert_to_fibgz, [('dwi_mask', 'mask_file')]),
        (convert_to_fibgz, outputnode, [('fibgz_file', 'fibgz')])
    ])  # fmt:skip

    if plot_reports:
        plot_peaks = pe.Node(CLIReconPeaksReport(), name="plot_peaks", n_procs=omp_nthreads)
        ds_report_peaks = pe.Node(
            DerivativesDataSink(
                datatype="figures",
                desc="NODDI",
                suffix="peaks",
            ),
            name="ds_report_peaks",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, plot_peaks, [('dwi_mask', 'mask_file')]),
            (convert_to_fibgz, plot_peaks, [('fibgz_file', 'fib_file')]),
            (noddi_fit, plot_peaks, [('icvf_image', 'background_image')]),
            (plot_peaks, ds_report_peaks, [('peak_report', 'in_file')]),
        ])  # fmt:skip

    if qsirecon_suffix:
        ds_fibgz = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwimap",
                extension="fib.gz",
                compress=True,
            ),
            name=f"ds_{qsirecon_suffix}_fibgz",
            run_without_submitting=True,
        )
        workflow.connect([(outputnode, ds_fibgz, [("fibgz", "in_file")])])

        ds_recon_scalars = pe.Node(
            ReconScalarsDataSink(dismiss_entities=["desc"]),
            name="ds_recon_scalars",
            run_without_submitting=True,
        )
        workflow.connect([(recon_scalars, ds_recon_scalars, [("scalar_info", "recon_scalars")])])

        ds_config = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                param="AMICOconfig",
                model="noddi",
                suffix="dwimap",
                compress=True,
            ),
            name="ds_noddi_config",
            run_without_submitting=True,
        )
        workflow.connect([(outputnode, ds_config, [("config_file", "in_file")])])

    workflow.__desc__ = desc

    return clean_datasinks(workflow, qsirecon_suffix)
