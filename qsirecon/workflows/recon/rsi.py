"""
Restriction Spectrum Imaging (RSI) Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import RSIReconScalars
from ...interfaces.reports import ScalarReport
from ...interfaces.rsi import RSI, RSI_OUTPUTS, describe_rsi_model
from ...utils.bids import clean_datasinks
from ...utils.boilerplate import build_documentation
from ...utils.rsi import RSI_URL, RSI_VERSION
from .utils import init_scalar_output_wf


def init_rsi_fit_wf(
    inputs_dict,
    name='rsi_recon',
    qsirecon_suffix='',
    params={},
):
    """Reconstruct RSI scalars with a port of the ABCD Study implementation.

    Notes
    -----
    Expects prior qsirecon outputs as inputs. The workflow outputnode exposes
    the isotropic (``*iso``), directional (``*dir``) and total (``*total``) signal
    components of the restricted, hindered and free water compartments
    (for example ``restrictediso``), along with ``l2norm`` and ``recon_scalars``.

    """
    workflow = Workflow(name=name)
    suffix_str = f' (outputs written to qsirecon-{qsirecon_suffix})' if qsirecon_suffix else ''
    workflow.__desc__ = (
        f'\n\n#### Restriction Spectrum Imaging{suffix_str}\n\n'
        'The restriction spectrum imaging (RSI) model (@white2013) was fit using a port of '
        f'the ABCD Study implementation (@hagler2019; RSIproc {RSI_VERSION}, {RSI_URL}). '
    )

    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ['odf_rois']),
        name='inputnode',
    )
    rsi_outputs = list(RSI_OUTPUTS)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=rsi_outputs + ['recon_scalars']),
        name='outputnode',
    )
    omp_nthreads = config.nipype.omp_nthreads

    # RSI does not produce ODF peaks, so there is no peaks report to skip
    params.pop('plot_reports', None)

    recon_scalars = pe.Node(
        RSIReconScalars(
            dismiss_entities=['desc'],
            qsirecon_suffix=qsirecon_suffix,
        ),
        name='recon_scalars',
        run_without_submitting=True,
    )
    rsi_fit = pe.Node(RSI(**params), name='recon_rsi')
    workflow.__desc__ += (
        describe_rsi_model(rsi_fit.inputs)
        + 'Isotropic, directional and total signal components were computed for each '
        'compartment from the norms of its degree 0, degree > 0 and all spherical harmonic '
        'coefficients, respectively. ' + build_documentation(rsi_fit) + ' '
    )

    workflow.connect([
        (inputnode, rsi_fit, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file'),
        ]),
        (rsi_fit, outputnode, [(output, output) for output in rsi_outputs]),
        (rsi_fit, recon_scalars, [
            (field, field)
            for output in rsi_outputs
            for field in (output, f'{output}_metadata')
        ]),
    ])  # fmt:skip

    if qsirecon_suffix:
        scalar_output_wf = init_scalar_output_wf()
        workflow.connect([
            (inputnode, scalar_output_wf, [('dwi_file', 'inputnode.source_file')]),
            (recon_scalars, scalar_output_wf, [('scalar_info', 'inputnode.scalar_configs')]),
        ])  # fmt:skip

        plot_scalars = pe.Node(
            ScalarReport(),
            name='plot_scalars',
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, plot_scalars, [
                ('acpc_preproc', 'underlay'),
                ('acpc_seg', 'dseg'),
                ('dwi_mask', 'mask_file'),
            ]),
            (recon_scalars, plot_scalars, [('scalar_info', 'scalar_metadata')]),
            (scalar_output_wf, plot_scalars, [('outputnode.scalar_files', 'scalar_maps')]),
            (scalar_output_wf, outputnode, [('outputnode.scalar_configs', 'recon_scalars')]),
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
