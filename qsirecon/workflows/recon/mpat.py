"""
MPAT bundle tractography
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_mpat_wf

"""

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

from ... import config
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.mpat import MPATDerivativesDataSink, MPATRecon
from ...utils.bids import clean_datasinks
from .pyafq import _parse_qsirecon_params_dict

# Settings of mpat.run() that a recon spec may set in its "mpat" section.
# n_proc comes from --omp-nthreads, and max_mem from --mem unless the spec sets it.
MPAT_PARAMS = frozenset(
    {
        'angle',
        'bundles',
        'double_odf',
        'fornix_exclude',
        'max_length',
        'max_mem',
        'midline_tol',
        'moments',
        'tol',
        'viz',
    }
)
# What MPAT may use of the memory QSIRecon was given: min(80%, limit - 10 GB),
# but at least half of it, as MPAT's own max_mem="auto" does with the job's limit.
_MEM_FRACTION, _MEM_HEADROOM_GB, _MEM_FLOOR = 0.8, 10.0, 0.5


def _mpat_memory_budget(memory_gb):
    """Memory budget for MPAT's solves, given QSIRecon's ``--mem`` (in GB), or "auto"."""
    if not memory_gb:
        return 'auto'

    budget = max(
        min(_MEM_FRACTION * memory_gb, memory_gb - _MEM_HEADROOM_GB),
        _MEM_FLOOR * memory_gb,
    )
    return f'{budget:.2f}GB'


def _mpat_node_mem_gb(memory_gb):
    """Memory the MPAT node reserves in Nipype's scheduler, in GB.

    MPAT sizes its solves to the memory it is given, so the node reserves all of it:
    QSIRecon's ``--mem``, or else most of the memory Nipype's MultiProc plugin manages
    (90% of the system's memory). Two MPAT nodes then never run at the same time.
    """
    if memory_gb:
        return memory_gb

    from nipype.utils.profiler import get_system_total_memory_gb

    return 0.85 * get_system_total_memory_gb()


def init_mpat_wf(inputs_dict, name='mpat', qsirecon_suffix='', params={}):
    """Segment white matter bundles with MPAT.

    MPAT (Markov probabilistic absorbing tractography) computes each bundle as the exact
    expectation over every streamline a probabilistic tracker could draw.
    pyAFQ supplies the fODF, the tissue partial volume estimates, the seeds,
    and the bundle ROIs warped into the subject.

    Parameters
    ----------
    inputs_dict : :obj:`dict`
        Information about the inputs to the workflow.
    name : :obj:`str`
        Name of the workflow.
    qsirecon_suffix : :obj:`str`
        Suffix of the derivatives directory the outputs are written to.
    params : :obj:`dict`
        Parameters from the recon spec: ``pyafq`` holds ParticipantAFQ settings
        (as in the ``pyafq_tractometry`` action), and ``mpat`` holds the settings of MPAT itself.

    Inputs
    ------
    *qsirecon outputs*

    Outputs
    -------
    mpat_dir
        Directory with the bundle density, endpoint, seed probability and visitation ODF maps,
        the bundle cores, and per-bundle summaries.
    """
    if not inputs_dict.get('has_qsiprep_anat', False):
        raise ValueError(
            'MPAT requires an anatomical image, brain mask and tissue probability maps aligned '
            'with the DWI data, but no high-resolution anatomical data were found in the '
            'preprocessed inputs.'
        )

    # plot_reports is added to every node by --skip-odf-reports; MPAT makes no ODF plots.
    unknown = sorted(set(params) - {'pyafq', 'mpat', 'plot_reports'})
    if unknown:
        raise ValueError(
            f'Unknown MPAT parameters: {", ".join(unknown)}. '
            'Put pyAFQ settings under "pyafq" and MPAT settings under "mpat".'
        )

    mpat_params = dict(params.get('mpat') or {})
    unknown = sorted(set(mpat_params) - MPAT_PARAMS)
    if unknown:
        raise ValueError(
            f'Unknown parameters in the "mpat" section: {", ".join(unknown)}. '
            f'Valid parameters are: {", ".join(sorted(MPAT_PARAMS))}.'
        )

    if mpat_params.get('max_mem') in (None, 'auto'):
        mpat_params['max_mem'] = _mpat_memory_budget(config.nipype.memory_gb)

    afq_kwargs = _parse_qsirecon_params_dict(params.get('pyafq') or {})

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['mpat_dir', 'recon_scalars']),
        name='outputnode',
    )
    outputnode.inputs.recon_scalars = []
    omp_nthreads = config.nipype.omp_nthreads

    run_mpat = pe.Node(
        MPATRecon(kwargs=afq_kwargs, n_procs=omp_nthreads, **mpat_params),
        name='run_mpat',
        n_procs=omp_nthreads,
        mem_gb=_mpat_node_mem_gb(config.nipype.memory_gb),
    )
    workflow.connect([
        (inputnode, run_mpat, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            # pyAFQ needs anatomical images registered to (not resampled onto) the DWI.
            # MPAT specs run in T1w space, so the QSIPrep anatomical outputs qualify.
            ('acpc_preproc', 't1_file'),
            ('acpc_brain_mask', 'mask_file'),
            ('acpc_csf_probseg', 'csf_probseg'),
            ('acpc_gm_probseg', 'gm_probseg'),
            ('acpc_wm_probseg', 'wm_probseg'),
        ]),
        (run_mpat, outputnode, [('mpat_dir', 'mpat_dir')]),
    ])  # fmt:skip

    suffix_str = f' (outputs written to qsirecon-{qsirecon_suffix})' if qsirecon_suffix else ''
    run = run_mpat.inputs
    odf = 'doubleODF' if run.double_odf else 'first-order (singleODF)'
    workflow.__desc__ = (
        f'\n\n#### MPAT{suffix_str}\n\n'
        'White matter bundles were computed with Markov probabilistic absorbing tractography '
        '[MPAT, @mpat], which treats probabilistic tractography as an absorbing Markov chain '
        'on white matter voxels and directions, and computes the expected density of each '
        'bundle over every streamline the tracker could draw, without sampling streamlines. '
        'Transition probabilities were derived from the analytic hop geometry of MITTENS '
        f'[@mittens] with {odf} transitions and a maximum turn of {run.angle:g} degrees '
        'per one-voxel hop. '
        'The fiber orientation distributions, white matter/gray matter interface seeds, '
        f'and bundle definitions came from pyAFQ [@pyafq; @pyafq2] with the settings '
        f'{afq_kwargs!s}, using the tissue probability maps from QSIPrep. '
        'Walkers moved through voxels where the white matter partial volume exceeded 0.4, '
        f'and solves stopped when the remaining walker mass fell below {run.tol:g}.'
    )

    if qsirecon_suffix:
        # Each MPAT output is written on its own, named from the preprocessed DWI
        # (subject, session, space, ...) plus the entities MPATRecon assigns it.
        ds_mpat = pe.MapNode(
            MPATDerivativesDataSink(dismiss_entities=['desc']),
            iterfield=['in_file', 'model', 'bundle', 'param', 'desc', 'suffix', 'compress'],
            name=f'ds_{name}',
            run_without_submitting=True,
        )
        workflow.connect([
            (run_mpat, ds_mpat, [
                ('out_files', 'in_file'),
                ('out_models', 'model'),
                ('out_bundles', 'bundle'),
                ('out_params', 'param'),
                ('out_descs', 'desc'),
                ('out_suffixes', 'suffix'),
                # Without it, the sink writes .nii.gz inputs as .nii
                ('out_compress', 'compress'),
            ]),
        ])  # fmt:skip

    if qsirecon_suffix and run_mpat.inputs.viz:
        # The collage and viewer pages go to the figures folder, with the reports.
        ds_figures = pe.MapNode(
            DerivativesDataSink(
                datatype='figures',
                model='mpat',
                suffix='dwimap',
                dismiss_entities=['desc'],
            ),
            iterfield=['in_file'],
            name=f'ds_{name}_figures',
            run_without_submitting=True,
        )
        ds_bundle_figures = pe.MapNode(
            DerivativesDataSink(
                datatype='figures',
                model='mpat',
                bundle='',
                suffix='dwimap',
                dismiss_entities=['desc'],
            ),
            iterfield=['in_file', 'bundle'],
            name=f'ds_{name}_bundle_figures',
            run_without_submitting=True,
        )
        workflow.connect([
            (run_mpat, ds_figures, [('figures', 'in_file')]),
            (run_mpat, ds_bundle_figures, [
                ('bundle_figures', 'in_file'),
                ('bundle_figure_bundles', 'bundle'),
            ]),
        ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)
