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
import re
import shutil

from nipype import logging
from nipype.interfaces.base import (
    File,
    SimpleInterface,
    TraitedSpec,
    Undefined,
    traits,
)

from .bids import DerivativesDataSink
from .pyafq import PyAFQBaseInputSpec, participant_afq_kwargs, shim_pyafq_inputs

LOGGER = logging.getLogger('nipype.interface')

# Entities that differ between MPAT's outputs. '' means the file does not use the entity.
MPAT_ENTITIES = ('model', 'bundle', 'param', 'desc')


class MPATDerivativesDataSink(DerivativesDataSink):
    """Write one MPAT output, leaving out the entities that are '' for it.

    MPAT's outputs use different sets of entities, so a MapNode over them gives every file
    every entity, with '' for the ones it does not use.
    """

    def _run_interface(self, runtime):
        for name in MPAT_ENTITIES:
            if getattr(self.inputs, name) == '':
                setattr(self.inputs, name, Undefined)

        return super()._run_interface(runtime)


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


# MPAT names its files <stem>_bundle-<Name>_desc-<what>[_<suffix>].<ext> and
# <stem>_desc-<what>_<suffix>.<ext>. These tables give the BIDS entities each one is
# written with, on top of the entities of the preprocessed DWI (subject, session, space, ...).
# Keys are (desc, suffix, extension) as MPAT writes them; None means "no suffix".
_BUNDLE_OUTPUTS = {
    ('density', 'map', '.nii.gz'): {'param': 'density', 'suffix': 'dwimap'},
    # MPAT's per-bundle summary (settings, solver diagnostics) describes the density map
    ('summary', None, '.json'): {'param': 'density', 'suffix': 'dwimap'},
    ('endpoints', 'map', '.nii.gz'): {'param': 'endpoints', 'suffix': 'dwimap'},
    ('seedprob', 'probmap', '.nii.gz'): {'param': 'seedprob', 'suffix': 'dwimap'},
    ('visitodf', 'dwimap', '.nii.gz'): {'param': 'visitodf', 'suffix': 'dwimap'},
    ('start', 'mask', '.nii.gz'): {'desc': 'start', 'suffix': 'mask'},
    ('end', 'mask', '.nii.gz'): {'desc': 'end', 'suffix': 'mask'},
    ('include', 'mask', '.nii.gz'): {'desc': 'include', 'suffix': 'mask'},
    ('exclude', 'mask', '.nii.gz'): {'desc': 'exclude', 'suffix': 'mask'},
    ('core', None, '.trx'): {'desc': 'core', 'suffix': 'streamlines'},
    ('core', None, '.json'): {'desc': 'core', 'suffix': 'streamlines'},
    # The core's nodes: coordinates and the density along the core
    ('core', 'path', '.tsv'): {'desc': 'core', 'suffix': 'bundlestats'},
}
_SUBJECT_OUTPUTS = {
    ('mpat', 'table', '.tsv'): {'suffix': 'bundlestats'},
    ('mpat', 'dseg', '.nii.gz'): {'desc': 'tissue', 'suffix': 'dseg'},
    ('overlength', 'map', '.nii.gz'): {'param': 'overlength', 'suffix': 'dwimap'},
}
_BUNDLE_RE = re.compile(
    r'_bundle-(?P<bundle>[A-Za-z0-9]+)_desc-(?P<desc>[A-Za-z0-9]+)'
    r'(?:_(?P<suffix>[a-z]+))?(?P<ext>\.[a-z]+(?:\.gz)?)$'
)
_SUBJECT_RE = re.compile(
    r'_desc-(?P<desc>[A-Za-z0-9]+)_(?P<suffix>[a-z]+)(?P<ext>\.[a-z]+(?:\.gz)?)$'
)
# Written only when a solve stalls: where the remaining walker mass sits
_TRAPPED_RE = re.compile(r'^trapped(?P<what>[a-z]+)(?P<n>\d+)$')
# Written only when a bundle's endpoints are SynthSeg labels
_SYNTHSEG_RE = re.compile(r'_model-(?P<model>synthseg2[a-z]+)_dseg\.nii\.gz$')


def collect_bids_outputs(mpat_dir):
    """Sort MPAT's output files into the derivatives QSIRecon writes.

    Parameters
    ----------
    mpat_dir : str
        Directory MPAT wrote its outputs to.

    Returns
    -------
    derivatives : list of dict
        One dictionary per file to write to the ``dwi`` folder: ``in_file`` and the BIDS
        entities ``model``, ``bundle``, ``param``, ``desc`` and ``suffix`` ('' when unused).
    bundle_figures : list of tuple
        (file, bundle) for each bundle's viewer page.
    figures : list of str
        The collage of all bundles and the viewer page with all of them.
    skipped : list of str
        Files that are not derivatives.
    """
    derivatives, bundle_figures, figures, skipped = [], [], [], []
    for fname in sorted(os.listdir(mpat_dir)):
        path = op.join(mpat_dir, fname)
        entities = None
        match = _BUNDLE_RE.search(fname)
        if match:
            key = (match['desc'], match['suffix'], match['ext'])
            if key == ('view', None, '.html'):
                bundle_figures.append((path, match['bundle']))
                continue

            if key in _BUNDLE_OUTPUTS:
                entities = dict(_BUNDLE_OUTPUTS[key], bundle=match['bundle'])
        elif match := _SUBJECT_RE.search(fname):
            key = (match['desc'], match['suffix'], match['ext'])
            trapped = _TRAPPED_RE.match(match['desc'])
            if key in (('mpat', 'collage', '.png'), ('mpat', 'view', '.html')):
                figures.append(path)
                continue

            if key in _SUBJECT_OUTPUTS:
                entities = dict(_SUBJECT_OUTPUTS[key])
            elif trapped and key[1:] == ('map', '.nii.gz'):
                entities = {
                    'param': 'trapped',
                    'desc': f'{trapped["what"]}{trapped["n"]}',
                    'suffix': 'dwimap',
                }
        elif match := _SYNTHSEG_RE.search(fname):
            entities = {'model': match['model'], 'suffix': 'dseg'}

        if entities is None:
            skipped.append(path)
            continue

        derivative = {'in_file': path, 'model': 'mpat', 'bundle': '', 'param': '', 'desc': ''}
        derivative.update(entities)
        derivatives.append(derivative)

    return derivatives, bundle_figures, figures, skipped


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
    out_files = traits.List(File(exists=True), desc='MPAT outputs to write as derivatives')
    out_models = traits.List(traits.Str(), desc='model entity of each file in out_files')
    out_bundles = traits.List(traits.Str(), desc="bundle entity ('' for none) of each file")
    out_params = traits.List(traits.Str(), desc="param entity ('' for none) of each file")
    out_descs = traits.List(traits.Str(), desc="desc entity ('' for none) of each file")
    out_suffixes = traits.List(traits.Str(), desc='suffix of each file in out_files')
    out_compress = traits.List(
        traits.Bool(),
        desc='whether each file in out_files is written gzipped (as MPAT wrote it)',
    )
    bundle_figures = traits.List(File(exists=True), desc="each bundle's viewer page")
    bundle_figure_bundles = traits.List(
        traits.Str(),
        desc='bundle entity of each file in bundle_figures',
    )
    figures = traits.List(
        File(exists=True),
        desc='collage of all bundles (PNG) and the viewer page with all of them (HTML)',
    )


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

        derivatives, bundle_figures, figures, skipped = collect_bids_outputs(mpat_dir)
        if skipped:
            LOGGER.warning(
                'These MPAT outputs are not written to the derivatives: %s',
                ', '.join(op.basename(f) for f in skipped),
            )

        self._results['mpat_dir'] = mpat_dir
        self._results['out_files'] = [d['in_file'] for d in derivatives]
        for entity in MPAT_ENTITIES:
            self._results[f'out_{entity}s'] = [d[entity] for d in derivatives]

        self._results['out_suffixes'] = [d['suffix'] for d in derivatives]
        self._results['out_compress'] = [d['in_file'].endswith('.gz') for d in derivatives]
        self._results['bundle_figures'] = [f for f, _ in bundle_figures]
        self._results['bundle_figure_bundles'] = [b for _, b in bundle_figures]
        self._results['figures'] = figures
        return runtime
