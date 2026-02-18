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
from ...interfaces.recon_scalars import AMICOReconScalars
from ...interfaces.reports import CLIReconPeaksReport, ScalarReport
from ...utils.bids import clean_datasinks
from ...utils.boilerplate import build_documentation
from ...utils.misc import load_yaml
from .utils import init_scalar_output_wf
from qsirecon.data import load as load_data


def init_amico_noddi_fit_wf(
    inputs_dict,
    name='amico_noddi_recon',
    qsirecon_suffix='',
    params={},
):
    """Reconstruct NODDI scalars using AMICO.

    Inputs

        *qsirecon outputs*

    Outputs

        directions
            Image of directions
        icvf
            Voxelwise ICVF.
        od
            Voxelwise Orientation Dispersion
        isovf
            Voxelwise ISOVF
        modulated_icvf
            Voxelwise modulated ICVF (ICVF * (1 - ISOVF))
        modulated_od
            Voxelwise modulated Orientation Dispersion  (OD * (1 - ISOVF))
        rmse
            Voxelwise root mean square error between predicted and measured signal
        nrmse
            Voxelwise normalized root mean square error between predicted and measured signal
        config
            Pickle file with model configurations in it
        fibgz

    """
    workflow = Workflow(name=name)
    suffix_str = f' (outputs written to qsirecon-{qsirecon_suffix})' if qsirecon_suffix else ''
    workflow.__desc__ = (
        f'\n\n#### NODDI Reconstruction{suffix_str}\n\n'
        'The NODDI model (@noddi) was fit using the AMICO implementation (@amico). '
    )

    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ['odf_rois']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'directions',
                'icvf',
                'od',
                'isovf',
                'modulated_icvf',
                'modulated_od',
                'rmse',
                'nrmse',
                'config',
                'fibgz',
                'recon_scalars',
                'tf',
            ],
        ),
        name='outputnode',
    )
    omp_nthreads = config.nipype.omp_nthreads

    plot_reports = params.pop('plot_reports', True)

    recon_scalars = pe.Node(
        AMICOReconScalars(
            dismiss_entities=['desc'],
            qsirecon_suffix=qsirecon_suffix,
        ),
        name='recon_scalars',
        run_without_submitting=True,
    )
    noddi_fit = pe.Node(NODDI(**params), name='recon_noddi', n_procs=omp_nthreads)
    workflow.__desc__ += build_documentation(noddi_fit) + ' '

    if params.get('saveModulatedMaps', True):
        noddi_tissue_fraction = pe.Node(NODDITissueFraction(), name='noddi_tissue_fraction')
        workflow.__desc__ += (
            'AMICO does not save the tissue fraction map. Therefore, '
            'the output tissue fraction map was separately reconstructed using '
            'custom Python code matching the AMICO implementation. '
        )
    convert_to_fibgz = pe.Node(NODDItoFIBGZ(), name='convert_to_fibgz')

    workflow.connect([
        (inputnode, noddi_fit, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file'),
        ]),
        (noddi_fit, outputnode, [
            ('directions', 'directions'),
            ('icvf', 'icvf'),
            ('od', 'od'),
            ('isovf', 'isovf'),
            ('config_file', 'config_file'),
        ]),
        (noddi_fit, recon_scalars, [
            ('icvf', 'icvf'),
            ('icvf_metadata', 'icvf_metadata'),
            ('od', 'od'),
            ('od_metadata', 'od_metadata'),
            ('isovf', 'isovf'),
            ('isovf_metadata', 'isovf_metadata'),
            ('directions', 'directions'),
            ('directions_metadata', 'directions_metadata'),
        ]),
        (noddi_fit, convert_to_fibgz, [
            ('directions', 'directions'),
            ('icvf', 'icvf'),
            ('od', 'od'),
            ('isovf', 'isovf'),
        ]),
        (inputnode, convert_to_fibgz, [('dwi_mask', 'mask_file')]),
        (convert_to_fibgz, outputnode, [('fibgz_file', 'fibgz')]),
    ])  # fmt:skip

    if params.get('saveModulatedMaps', True):
        workflow.connect([
            (inputnode, noddi_tissue_fraction, [("dwi_mask", "mask_image")]),
            (noddi_fit, noddi_tissue_fraction, [("isovf", "isovf")]),
            (noddi_tissue_fraction, outputnode, [("tf", "tf")]),
            (noddi_tissue_fraction, recon_scalars, [("tf", "tf")]),
            (noddi_fit, recon_scalars, [("isovf_metadata", "tf_metadata")]),
            (noddi_fit, recon_scalars, [
                ("modulated_icvf", "modulated_icvf"),
                ("modulated_icvf_metadata", "modulated_icvf_metadata"),
                ("modulated_od", "modulated_od"),
                ("modulated_od_metadata", "modulated_od_metadata"),
            ]),
            (noddi_fit, convert_to_fibgz, [
                ("modulated_icvf", "modulated_icvf"),
                ("modulated_od", "modulated_od"),
            ]),
        ])  # fmt:skip

    if params.get('rmse', True):
        workflow.connect([
            (noddi_fit, recon_scalars, [
                ("rmse", "rmse"),
                ("rmse_metadata", "rmse_metadata"),
            ]),
        ])  # fmt:skip

    if params.get('nrmse', True):
        workflow.connect([
            (noddi_fit, recon_scalars, [
                ("nrmse", "nrmse"),
                ("nrmse_metadata", "nrmse_metadata"),
            ]),
        ])  # fmt:skip

    if plot_reports:
        plot_peaks = pe.Node(
            CLIReconPeaksReport(),
            name='plot_peaks',
            n_procs=omp_nthreads,
        )
        ds_report_peaks = pe.Node(
            DerivativesDataSink(
                datatype='figures',
                desc='NODDI',
                suffix='peaks',
            ),
            name='ds_report_peaks',
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, plot_peaks, [('dwi_mask', 'mask_file')]),
            (convert_to_fibgz, plot_peaks, [('fibgz_file', 'fib_file')]),
            (noddi_fit, plot_peaks, [('icvf', 'background_image')]),
            (plot_peaks, ds_report_peaks, [('peak_report', 'in_file')]),
        ])  # fmt:skip

    if qsirecon_suffix:
        derivatives_config = load_yaml(load_data('nonscalars/amico_noddi.yaml'))
        ds_fibgz = pe.Node(
            DerivativesDataSink(
                dismiss_entities=['desc'],
                compress=True,
                **derivatives_config['fibgz']['bids'],
            ),
            name=f'ds_{qsirecon_suffix}_fibgz',
            run_without_submitting=True,
        )
        workflow.connect([(outputnode, ds_fibgz, [('fibgz', 'in_file')])])

        scalar_output_wf = init_scalar_output_wf()
        workflow.connect([
            (inputnode, scalar_output_wf, [("dwi_file", "inputnode.source_file")]),
            (recon_scalars, scalar_output_wf, [("scalar_info", "inputnode.scalar_configs")]),
        ])  # fmt:skip

        ds_config = pe.Node(
            DerivativesDataSink(
                dismiss_entities=['desc'],
                compress=True,
                **derivatives_config['config_file']['bids'],
            ),
            name='ds_noddi_config',
            run_without_submitting=True,
        )
        workflow.connect([(outputnode, ds_config, [('config_file', 'in_file')])])

        plot_scalars = pe.Node(
            ScalarReport(),
            name='plot_scalars',
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
                datatype='figures',
                desc='scalars',
                suffix='dwimap',
                dismiss_entities=['dsistudiotemplate'],
            ),
            name='ds_report_scalars',
            run_without_submitting=True,
        )
        workflow.connect([(plot_scalars, ds_report_scalars, [('out_report', 'in_file')])])
    else:
        # If not writing out scalar files, pass the working directory scalar configs
        workflow.connect([(recon_scalars, outputnode, [('scalar_info', 'recon_scalars')])])

    return clean_datasinks(workflow, qsirecon_suffix)
