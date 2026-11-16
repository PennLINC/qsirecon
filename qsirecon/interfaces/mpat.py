# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interface for running MPAT
~~~~~~~~~~~~~~~~~~~~~~~~~~

MPAT (Markov probabilistic absorbing tractography) computes white matter bundles as exact
expectations over every streamline a probabilistic tracker could draw, without sampling.
It runs on top of a pyAFQ ``ParticipantAFQ``, which supplies the fODF, the tissue partial
volume estimates, the seeds and the bundle ROIs warped into the subject.
"""

import os
import os.path as op
import shutil

from nipype.interfaces.base import (
    SimpleInterface,
    TraitedSpec,
    traits,
)

from .pyafq import PyAFQBaseInputSpec, participant_afq_kwargs, shim_pyafq_inputs


def default_bundle_dict():
    """The bundles MPAT computes in QSIRecon when the recon spec does not name any.

    These are pyAFQ's default, SLF and callosal bundles, whose templates the QSIRecon image
    caches. MPAT's own default (``mpat.bundles.mpat_bd``) adds the hippocampal cingulum and
    the fornix, whose endpoints are SynthSeg labels. SynthSeg needs onnxruntime and model
    weights that are downloaded at run time, so QSIRecon leaves those two out.
    MPAT's rules for pyAFQ's bundles (``mpat.bundles.MPAT_RULES``) apply either way.
    """
    from AFQ.api.bundle_dict import callosal_bd, default_bd, slf_bd

    return default_bd() + slf_bd() + callosal_bd()


class _MPATInputSpec(PyAFQBaseInputSpec):
    max_mem = traits.Either(
        traits.Float(),
        traits.Str(),
        default='auto',
        usedefault=True,
        desc=(
            'memory budget for the solves, in bytes or as a string such as "16GB", '
            'or "auto" to let MPAT read the memory limit of the job'
        ),
    )
    bundles = traits.Either(
        None,
        traits.List(traits.Str()),
        default=None,
        usedefault=True,
        desc='names of the bundles to compute (default: every bundle in the bundle dictionary)',
    )
    moments = traits.Bool(
        False,
        usedefault=True,
        desc='track spatial first moments (narrower bundles, about five times slower)',
    )
    tol = traits.Float(
        1e-6,
        usedefault=True,
        desc='remaining walker mass at which the solves stop',
    )
    angle = traits.Float(
        35.0,
        usedefault=True,
        desc='maximum turn per one-voxel hop, in degrees',
    )
    double_odf = traits.Bool(
        True,
        usedefault=True,
        desc=(
            'weight each hop by the fODF where it lands (doubleODF); False gives first-order hops'
        ),
    )
    max_length = traits.Float(
        250.0,
        usedefault=True,
        desc='soft maximum streamline length in mm (0 turns it off)',
    )
    midline_tol = traits.Either(
        None,
        traits.Float(),
        default=None,
        usedefault=True,
        desc=(
            'distance in mm past the midline the corticospinal tracts may go before the other '
            "hemisphere is excluded (default: MPAT's rule, 10 mm)"
        ),
    )
    fornix_exclude = traits.Bool(
        True,
        usedefault=True,
        desc='exclude the fornix and septum from the anterior thalamic radiations',
    )
    viz = traits.Bool(
        True,
        usedefault=True,
        desc='write a collage of the bundles (PNG) and standalone HTML viewer pages',
    )


class _MPATOutputSpec(TraitedSpec):
    mpat_dir = traits.Directory(exists=True, desc='directory with the MPAT outputs')


class MPATRecon(SimpleInterface):
    """Segment pyAFQ's bundles with MPAT."""

    input_spec = _MPATInputSpec
    output_spec = _MPATOutputSpec

    def _run_interface(self, runtime):
        import mpat
        from AFQ.api.participant import ParticipantAFQ

        shim_dir, files = shim_pyafq_inputs(
            self.inputs,
            runtime.cwd,
            (
                'bval_file',
                'bvec_file',
                'dwi_file',
                't1_file',
                'mask_file',
                'csf_probseg',
                'gm_probseg',
                'wm_probseg',
            ),
        )
        kwargs = participant_afq_kwargs(self.inputs.kwargs, files, self.inputs.n_procs)
        # MPAT walks where the white matter partial volume exceeds 0.4, so it needs the
        # tissue probability maps. Without them pyAFQ would run SynthSeg on the T1w.
        if 'pve' not in kwargs:
            raise ValueError(
                'MPAT requires CSF, GM and WM probability maps aligned with the T1w image '
                '(the QSIPrep label-{CSF,GM,WM}_probseg outputs), but at least one is missing.'
            )

        if kwargs.get('bundle_info') is None:
            kwargs['bundle_info'] = default_bundle_dict()

        tracking_params = dict(kwargs.get('tracking_params') or {})
        tracking_params.setdefault('odf_model', 'CSD')
        kwargs['tracking_params'] = tracking_params

        afq_dir = op.join(shim_dir, 'PYAFQ')
        os.makedirs(afq_dir, exist_ok=True)
        afq = ParticipantAFQ(
            files['dwi_file'],
            files['bval_file'],
            files['bvec_file'],
            files['t1_file'],
            afq_dir,
            **kwargs,
        )

        mpat_dir = op.join(runtime.cwd, 'mpat')
        # Every file in this directory becomes a derivative, so start from an empty one.
        shutil.rmtree(mpat_dir, ignore_errors=True)
        mpat.run(
            afq,
            n_proc=self.inputs.n_procs,
            max_mem=self.inputs.max_mem,
            bundles=self.inputs.bundles,
            moments=self.inputs.moments,
            tol=self.inputs.tol,
            angle=self.inputs.angle,
            double_odf=self.inputs.double_odf,
            viz=self.inputs.viz,
            max_length=self.inputs.max_length,
            midline_tol=self.inputs.midline_tol,
            fornix=self.inputs.fornix_exclude,
            out_dir=mpat_dir,
        )
        # The cached transition matrix is a large work file, not a derivative.
        for fname in os.listdir(mpat_dir):
            if fname.endswith('_desc-chain_cache.npz'):
                os.remove(op.join(mpat_dir, fname))

        self._results['mpat_dir'] = mpat_dir
        return runtime
