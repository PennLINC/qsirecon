"""
QSIPrep QC maps as scalars
^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.qc_maps import (
    DEFAULT_QC_MAPS,
    QC_MAP_INPUTS,
    QSIPREP_QC_MAP_PATTERNS,
    QSIPrepQCReconScalars,
)
from ...interfaces.reports import ScalarReport
from ...utils.bids import clean_datasinks
from .utils import init_scalar_output_wf


def init_qsiprep_qc_scalars_wf(
    inputs_dict,
    name='qsiprep_qc_scalars',
    qsirecon_suffix='',
    params={},
):
    """Make scalar maps from the voxelwise QC images QSIPrep wrote.

    The scalars can be summarized by ``bundle_map``, ``template_map`` and
    ``parcellate_scalars`` nodes, like the scalars of a model.

    Parameters
    ----------
    params : dict
        ``include`` lists which maps to use, from ``noise``, ``snr``, ``tsnr``,
        ``cnr``, ``jacobian``, ``sdc``, ``rank`` and ``rankdensity``.
        By default, all but ``rank`` and ``rankdensity``.

    Notes
    -----
    The workflow outputnode exposes ``recon_scalars``.
    QC maps QSIPrep did not write are skipped.

    """
    include = params.get('include', DEFAULT_QC_MAPS)
    unknown = sorted(set(include) - set(QC_MAP_INPUTS))
    if unknown:
        raise ValueError(
            f'Unknown QSIPrep QC maps {unknown} in {name}. Choose from {", ".join(QC_MAP_INPUTS)}.'
        )

    workflow = Workflow(name=name)
    suffix_str = f' (outputs written to qsirecon-{qsirecon_suffix})' if qsirecon_suffix else ''
    workflow.__desc__ = (
        f'\n\n#### QSIPrep quality-control maps{suffix_str}\n\n'
        'The voxelwise quality-control maps written by QSIPrep '
        f'({", ".join(include)}) were used as scalar maps on the grid of the preprocessed '
        'DWI. Maps with one volume per input series or per shell were reduced to 3D: '
        'noise maps were pooled across series, weighted by the number of volumes in '
        'each series, and the CNR maps from eddy were split by shell. '
        'The signal-to-noise ratio was computed as the b=0 reference divided by the '
        'noise standard deviation, and the susceptibility distortion correction '
        'displacement was summarized by its magnitude in mm. '
    )

    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['recon_scalars']), name='outputnode')
    omp_nthreads = config.nipype.omp_nthreads

    recon_scalars = pe.Node(
        QSIPrepQCReconScalars(
            dismiss_entities=['desc'],
            qsirecon_suffix=qsirecon_suffix,
            include=include,
        ),
        name='recon_scalars',
    )
    workflow.connect([
        (inputnode, recon_scalars, [
            (field, field)
            for field in [*QSIPREP_QC_MAP_PATTERNS, 'dwi_ref', 'dwi_mask', 'bval_file']
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
