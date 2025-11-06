"""
AMICO Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_amico_noddi_fit_wf

"""

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.amico import NODDI, NODDITissueFraction
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.converters import NODDItoFIBGZ
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import create_recon_scalars_class
from ...interfaces.reports import CLIReconPeaksReport, ScalarReport
from ...utils.bids import clean_datasinks
from ...utils.misc import load_yaml
from .utils import init_scalar_output_wf
from qsirecon.data import load as load_data


def init_amico_noddi_fit_wf(
    inputs_dict,
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
        modulated_icvf_image
            Voxelwise modulated ICVF (ICVF * (1 - ISOVF))
        modulated_od_image
            Voxelwise modulated Orientation Dispersion  (OD * (1 - ISOVF))
        rmse_image
            Voxelwise root mean square error between predicted and measured signal
        nrmse_image
            Voxelwise normalized root mean square error between predicted and measured signal
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
                "modulated_icvf_image",
                "modulated_od_image",
                "rmse_image",
                "nrmse_image",
                "config_file",
                "fibgz",
                "recon_scalars",
                "tf_image",
            ],
        ),
        name="outputnode",
    )
    omp_nthreads = config.nipype.omp_nthreads
    workflow = Workflow(name=name)

    plot_reports = params.pop("plot_reports", True)
    desc = """
### NODDI Reconstruction

"""
    desc += """\
The NODDI model (@noddi) was fit using the AMICO implementation (@amico).
A value of %.1E was used for parallel diffusivity and %.1E for isotropic
diffusivity.""" % (
        params["dPar"],
        params["dIso"],
    )
    if params.get("is_exvivo"):
        desc += " An additional component was added to the model for ex-vivo data."

    desc += """\
 Tissue fraction (1 - ISOVF) modulated ICVF and Orientation Dispersion maps
were also computed (@parker2021not)."""

    recon_scalars_class = create_recon_scalars_class(load_data("scalars/amico_noddi.yaml"))
    recon_scalars = pe.Node(
        recon_scalars_class(dismiss_entities=["desc"], qsirecon_suffix=qsirecon_suffix),
        name="recon_scalars",
        run_without_submitting=True,
    )
    noddi_fit = pe.Node(NODDI(**params), name="recon_noddi", n_procs=omp_nthreads)
    noddi_tissue_fraction = pe.Node(NODDITissueFraction(), name="noddi_tissue_fraction")
    convert_to_fibgz = pe.Node(NODDItoFIBGZ(), name="convert_to_fibgz")

    workflow.connect([
        (inputnode, noddi_fit, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file'),
        ]),
        (inputnode, noddi_tissue_fraction, [('dwi_mask', 'mask_image')]),
        (noddi_fit, noddi_tissue_fraction, [
            ('isovf_image', 'isovf_image'),
        ]),
        (noddi_tissue_fraction, outputnode, [('tf_image', 'tf_image')]),
        (noddi_tissue_fraction, recon_scalars, [('tf_image', 'tf_image')]),
        (noddi_fit, outputnode, [
            ('directions_image', 'directions_image'),
            ('icvf_image', 'icvf_image'),
            ('od_image', 'od_image'),
            ('isovf_image', 'isovf_image'),
            ('modulated_icvf_image', 'modulated_icvf_image'),
            ('modulated_od_image', 'modulated_od_image'),
            ('rmse_image', 'rmse_image'),
            ('nrmse_image', 'nrmse_image'),
            ('config_file', 'config_file'),
        ]),
        (noddi_fit, recon_scalars, [
            ('icvf_image', 'icvf_image'),
            ('od_image', 'od_image'),
            ('isovf_image', 'isovf_image'),
            ('directions_image', 'directions_image'),
            ('modulated_icvf_image', 'modulated_icvf_image'),
            ('modulated_od_image', 'modulated_od_image'),
            ('rmse_image', 'rmse_image'),
            ('nrmse_image', 'nrmse_image'),
        ]),
        (noddi_fit, convert_to_fibgz, [
            ('directions_image', 'directions_file'),
            ('icvf_image', 'icvf_file'),
            ('od_image', 'od_file'),
            ('isovf_image', 'isovf_file'),
            ('modulated_icvf_image', 'modulated_icvf_file'),
            ('modulated_od_image', 'modulated_od_file'),
        ]),
        (inputnode, convert_to_fibgz, [('dwi_mask', 'mask_file')]),
        (convert_to_fibgz, outputnode, [('fibgz_file', 'fibgz')])
    ])  # fmt:skip

    if plot_reports:
        plot_peaks = pe.Node(
            CLIReconPeaksReport(),
            name="plot_peaks",
            n_procs=omp_nthreads,
        )
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
        derivatives_config = load_yaml(load_data("nonscalars/amico_noddi.yaml"))
        ds_fibgz = pe.Node(
            DerivativesDataSink(
                dismiss_entities=["desc"],
                compress=True,
                **derivatives_config["fibgz"]["bids"],
            ),
            name=f"ds_{qsirecon_suffix}_fibgz",
            run_without_submitting=True,
        )
        workflow.connect([(outputnode, ds_fibgz, [("fibgz", "in_file")])])

        scalar_output_wf = init_scalar_output_wf()
        workflow.connect([
            (inputnode, scalar_output_wf, [("dwi_file", "inputnode.source_file")]),
            (recon_scalars, scalar_output_wf, [("scalar_info", "inputnode.scalar_configs")]),
        ])  # fmt:skip

        ds_config = pe.Node(
            DerivativesDataSink(
                dismiss_entities=["desc"],
                compress=True,
                **derivatives_config["config_file"]["bids"],
            ),
            name="ds_noddi_config",
            run_without_submitting=True,
        )
        workflow.connect([(outputnode, ds_config, [("config_file", "in_file")])])

        plot_scalars = pe.Node(
            ScalarReport(),
            name="plot_scalars",
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, plot_scalars, [
                ("acpc_preproc", "underlay"),
                ("acpc_seg", "dseg"),
                ("dwi_mask", "mask_file"),
            ]),
            (recon_scalars, plot_scalars, [("scalar_info", "scalar_metadata")]),
            (scalar_output_wf, plot_scalars, [("outputnode.scalar_files", "scalar_maps")]),
            (scalar_output_wf, outputnode, [("outputnode.scalar_configs", "recon_scalars")]),
        ])  # fmt:skip

        ds_report_scalars = pe.Node(
            DerivativesDataSink(
                datatype="figures",
                desc="scalars",
                suffix="dwimap",
                dismiss_entities=["dsistudiotemplate"],
            ),
            name="ds_report_scalars",
            run_without_submitting=True,
        )
        workflow.connect([(plot_scalars, ds_report_scalars, [("out_report", "in_file")])])
    else:
        # If not writing out scalar files, pass the working directory scalar configs
        workflow.connect([(recon_scalars, outputnode, [("scalar_info", "recon_scalars")])])

    workflow.__desc__ = desc

    return clean_datasinks(workflow, qsirecon_suffix)
