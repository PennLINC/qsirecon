# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Use QSIPrep's voxelwise QC images as scalar maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QSIPrep writes several quality-control images on the grid of the preprocessed DWI:
the denoiser's noise map, b=0 tSNR, CNR, Jacobian weights and the susceptibility
distortion correction displacement.
:class:`QSIPrepQCReconScalars` turns them into 3D scalar maps that ``bundle_map``,
``template_map`` and ``parcellate_scalars`` can summarize like any model's scalars.
"""

import json
import os.path as op
import re
from copy import deepcopy
from glob import glob

import nibabel as nb
import numpy as np
from nibabel.processing import resample_from_to
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import split_filename

from qsirecon.data import load as load_data

from ..utils.misc import load_yaml

LOGGER = logging.getLogger('nipype.interface')

# Glob patterns for QSIPrep's QC images, relative to the preprocessed series name
# with its ``desc-preproc_dwi`` removed.
# These are QSIPrepDWIIngress outputs, so they reach every recon workflow.
QSIPREP_QC_MAP_PATTERNS = {
    'noise_map_file': 'stat-*_desc-noise_dwimap.nii*',
    'tsnr_file': 'stat-tsnr_dwimap.nii*',
    'cnr_file': 'model-*_stat-cnr_dwimap.nii*',
    'jacobian_file': 'desc-jacobian_dwimap.nii*',
    'sdc_displacement_file': 'desc-sdc_displacement.nii*',
    'sdc_refinement_file': 'desc-sdcrefinement_displacement.nii*',
    'rank_file': 'desc-rank_dwimap.nii*',
    'rankdensity_file': 'desc-rankdensity_dwimap.nii*',
}

# The QSIPrep images each ``include`` keyword of the ``qsiprep_qc_scalars`` action uses
QC_MAP_INPUTS = {
    'noise': ['noise_map_file'],
    'snr': ['noise_map_file'],
    'tsnr': ['tsnr_file'],
    'cnr': ['cnr_file'],
    'jacobian': ['jacobian_file'],
    'sdc': ['sdc_displacement_file', 'sdc_refinement_file'],
    'rank': ['rank_file'],
    'rankdensity': ['rankdensity_file'],
}
DEFAULT_QC_MAPS = ['noise', 'snr', 'tsnr', 'cnr', 'jacobian', 'sdc']

qsiprep_qc_scalars = load_yaml(load_data('scalars/qsiprep_qc.yaml'))

PREPROC_SUFFIX = 'desc-preproc_dwi'
# Sidecar keys that give, for each DWI volume, the volume of a 4D map that describes it
INDEX_KEYS = {
    'jacobian': 'JacobianWeightIndex',
    'noiserank': 'RankMapIndex',
    'noiserankdensity': 'RankDensityMapIndex',
}
# Volumes at or below this b-value are b=0 volumes when splitting CNR by shell
B0_THRESHOLD = 50
# b-values within this distance belong to the same shell
SHELL_DISTANCE = 50


def find_qsiprep_qc_maps(dwi_file):
    """Find the QC images QSIPrep wrote alongside a preprocessed DWI series.

    Returns a dict mapping each :data:`QSIPREP_QC_MAP_PATTERNS` key with exactly
    one match to its path.
    """
    out_root, fname, _ = split_filename(dwi_file)
    if not fname.endswith(PREPROC_SUFFIX):
        return {}

    prefix = op.join(out_root, fname[: -len(PREPROC_SUFFIX)])
    found = {}
    for name, pattern in QSIPREP_QC_MAP_PATTERNS.items():
        matches = glob(prefix + pattern)
        if len(matches) == 1:
            found[name] = matches[0]
        elif len(matches) > 1:
            LOGGER.warning(f'Ignoring {name}: more than one file matches {prefix + pattern}')
    return found


def usable_qc_maps(qc_files, include):
    """Select the QC images that will produce scalars.

    Parameters
    ----------
    qc_files : dict
        Paths keyed by :data:`QSIPREP_QC_MAP_PATTERNS` names.
    include : list of str
        Keywords from :data:`QC_MAP_INPUTS`.

    Returns
    -------
    dict
        The subset of ``qc_files`` to use.
        An all-zero tSNR map, which QSIPrep writes when there are fewer
        than two b=0 volumes, is left out.
    """
    wanted = {name for keyword in include for name in QC_MAP_INPUTS[keyword]}
    usable = {name: path for name, path in qc_files.items() if name in wanted}
    if 'tsnr_file' in usable and not np.any(np.asanyarray(nb.load(usable['tsnr_file']).dataobj)):
        LOGGER.info(f'Skipping {usable["tsnr_file"]}: it is all zeros')
        del usable['tsnr_file']
    return usable


class _QSIPrepQCReconScalarsInputSpec(BaseInterfaceInputSpec):
    source_file = File(
        exists=True,
        mandatory=True,
        desc='The preprocessed DWI. Scalars are written on its grid.',
    )
    qsirecon_suffix = traits.Str(mandatory=True, desc='Derivatives dataset for the scalars')
    dismiss_entities = traits.List([], usedefault=True)
    include = traits.List(
        traits.Enum(*QC_MAP_INPUTS),
        DEFAULT_QC_MAPS,
        usedefault=True,
        desc='Which QC maps to turn into scalars',
    )
    dwi_ref = File(exists=True, desc='Preprocessed b=0 reference, the numerator of the SNR')
    dwi_mask = File(exists=True, desc='Brain mask that limits the SNR map')
    bval_file = File(exists=True, desc='b-values of the preprocessed DWI, for per-shell CNR')


for _qc_map_name in QSIPREP_QC_MAP_PATTERNS:
    _QSIPrepQCReconScalarsInputSpec.add_class_trait(_qc_map_name, File(exists=True))


class _QSIPrepQCReconScalarsOutputSpec(TraitedSpec):
    scalar_info = traits.List()


class QSIPrepQCReconScalars(SimpleInterface):
    """Make 3D scalar maps on the DWI grid from QSIPrep's QC images.

    The 4D maps are reduced to 3D. Each map is resampled linearly if it is not on
    the grid of ``source_file``. The output, ``scalar_info``, has the same form as
    the output of :class:`~qsirecon.interfaces.recon_scalars.ReconScalars`.
    """

    input_spec = _QSIPrepQCReconScalarsInputSpec
    output_spec = _QSIPrepQCReconScalarsOutputSpec
    scalar_metadata = qsiprep_qc_scalars

    def _run_interface(self, runtime):
        from .recon_scalars import build_scalar_info, source_file_entities

        inputs = self.inputs
        qc_files = {
            name: getattr(inputs, name)
            for name in QSIPREP_QC_MAP_PATTERNS
            if isdefined(getattr(inputs, name))
        }
        qc_files = usable_qc_maps(qc_files, inputs.include)

        reference = nb.load(inputs.source_file)
        self._grid = (reference.shape[:3], reference.affine)
        self._cwd = runtime.cwd

        scalars = []
        if 'noise_map_file' in qc_files:
            scalars.extend(self._noise_scalars(qc_files['noise_map_file']))
        if 'tsnr_file' in qc_files:
            scalars.append(self._passthrough('b0tsnr', qc_files['tsnr_file']))
        if 'cnr_file' in qc_files:
            scalars.extend(self._cnr_scalars(qc_files['cnr_file']))
        if 'jacobian_file' in qc_files:
            scalars.append(self._passthrough('jacobian', qc_files['jacobian_file']))
        for variable_name, name in (
            ('sdcdisplacement', 'sdc_displacement_file'),
            ('sdcrefinement', 'sdc_refinement_file'),
        ):
            if name in qc_files:
                scalars.append(self._displacement_scalar(variable_name, qc_files[name]))
        if 'rank_file' in qc_files:
            scalars.append(self._passthrough('noiserank', qc_files['rank_file']))
        if 'rankdensity_file' in qc_files:
            scalars.append(self._passthrough('noiserankdensity', qc_files['rankdensity_file']))

        if not scalars:
            raise RuntimeError(
                f'None of the QSIPrep QC maps requested ({", ".join(inputs.include)}) '
                f'could be made into scalars for {inputs.source_file}'
            )

        source_file_bids = source_file_entities(inputs.source_file, inputs.dismiss_entities)
        self._results['scalar_info'] = [
            build_scalar_info(
                definition,
                path,
                variable_name=variable_name,
                qsirecon_suffix=inputs.qsirecon_suffix,
                source_file=inputs.source_file,
                source_file_bids=source_file_bids,
                metadata=metadata,
            )
            for variable_name, definition, path, metadata in scalars
        ]
        return runtime

    def _noise_scalars(self, noise_file):
        img = nb.load(noise_file)
        sidecar = _load_sidecar(noise_file)
        quantity = sidecar.get('Quantity')
        if quantity is None:
            quantity = 'residual_rms' if '_stat-rms_' in op.basename(noise_file) else 'sigma'
        variable_name = 'noisesigma' if quantity == 'sigma' else 'noiserms'

        data = img.get_fdata(dtype=np.float32)
        steps = []
        if data.ndim == 4:
            # Pool the variance of the series, weighted by how many volumes each describes
            weights = _volume_weights(sidecar.get('NoiseMapIndex'), data.shape[3])
            data = np.sqrt(np.tensordot(data**2, weights, axes=([3], [0])))
            steps.append(
                f'The {img.shape[3]} series maps were pooled as sqrt(sum(n_i * x_i^2) / '
                'sum(n_i)), where n_i is the number of volumes each describes (NoiseMapIndex)'
            )

        noise_img, resampled = self._on_grid(nb.Nifti1Image(data, img.affine))
        steps += resampled
        scalars = []
        if 'noise' in self.inputs.include:
            scalars.append(
                self._scalar(
                    variable_name,
                    noise_img if steps else noise_file,
                    noise_file,
                    sidecar,
                    steps,
                    write=bool(steps),
                )
            )
        if 'snr' in self.inputs.include:
            if quantity != 'sigma':
                LOGGER.info('Skipping SNR: the noise map is not a noise standard deviation')
            elif not isdefined(self.inputs.dwi_ref):
                LOGGER.info('Skipping SNR: there is no b=0 reference')
            else:
                scalars.append(self._snr_scalar(noise_img, noise_file, sidecar, steps))
        return scalars

    def _snr_scalar(self, noise_img, noise_file, sidecar, steps):
        b0_img, b0_resampled = self._on_grid(nb.load(self.inputs.dwi_ref))
        b0 = b0_img.get_fdata(dtype=np.float32)
        sigma = noise_img.get_fdata(dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = np.where(sigma > 0, b0 / sigma, 0)
        snr[~np.isfinite(snr)] = 0
        if isdefined(self.inputs.dwi_mask):
            mask_img, _ = self._on_grid(nb.load(self.inputs.dwi_mask), order=0)
            snr[np.asanyarray(mask_img.dataobj) == 0] = 0

        snr_steps = steps + [f'b=0 reference {step[0].lower()}{step[1:]}' for step in b0_resampled]
        snr_steps.append('The b=0 reference was divided by the noise standard deviation')
        return self._scalar(
            'snr',
            nb.Nifti1Image(snr.astype(np.float32), noise_img.affine),
            noise_file,
            sidecar,
            snr_steps,
            write=True,
        )

    def _cnr_scalars(self, cnr_file):
        img = nb.load(cnr_file)
        sidecar = _load_sidecar(cnr_file)
        match = re.search(r'_model-([a-zA-Z0-9]+)_', op.basename(cnr_file))
        description = self.scalar_metadata['cnr']['metadata']['Description']
        if match:
            description += f' ({match.group(1)} model)'
        if img.ndim == 3 or img.shape[3] == 1:
            return [self._passthrough('cnr', cnr_file, extra={'Description': description})]

        # eddy's --cnr_maps: the b=0 tSNR, then the CNR of each shell in ascending order
        shells = _shells(self.inputs.bval_file) if isdefined(self.inputs.bval_file) else []
        if img.shape[3] != len(shells) + 1:
            LOGGER.warning(
                f'Skipping {cnr_file}: its {img.shape[3]} volumes do not match a b=0 tSNR '
                f'map plus one CNR map for each of the {len(shells)} shells'
            )
            return []

        data = img.get_fdata(dtype=np.float32)
        scalars = []
        for i_shell, bval in enumerate(shells, start=1):
            shell_img, resampled = self._on_grid(nb.Nifti1Image(data[..., i_shell], img.affine))
            scalars.append(
                self._scalar(
                    f'cnr_b{bval}',
                    shell_img,
                    cnr_file,
                    sidecar,
                    [f'Volume {i_shell} of the eddy CNR maps, for the b={bval} shell', *resampled],
                    write=True,
                    definition='cnr',
                    desc=f'b{bval}',
                    extra={'Description': f'{description}, b={bval} shell'},
                )
            )
        return scalars

    def _displacement_scalar(self, variable_name, displacement_file):
        img = nb.load(displacement_file)
        sidecar = _load_sidecar(displacement_file)
        vectors = np.asanyarray(img.dataobj).astype(np.float32)
        vectors = vectors.reshape(img.shape[:3] + (-1,))
        if vectors.shape[3] != 3:
            raise ValueError(f'{displacement_file} is not a field of 3D displacement vectors')
        magnitude = np.linalg.norm(vectors, axis=3)
        magnitude_img, resampled = self._on_grid(nb.Nifti1Image(magnitude, img.affine))
        return self._scalar(
            variable_name,
            magnitude_img,
            displacement_file,
            sidecar,
            ['The Euclidean norm of the displacement vectors', *resampled],
            write=True,
        )

    def _passthrough(self, variable_name, qc_file, extra=None):
        """Use a QC map as it is, if it is 3D and on the DWI grid."""
        img = nb.load(qc_file)
        sidecar = _load_sidecar(qc_file)
        steps = []
        data = None
        if img.ndim == 4 and img.shape[3] > 1:
            index_key = INDEX_KEYS.get(variable_name)
            weights = _volume_weights(sidecar.get(index_key), img.shape[3])
            data = np.tensordot(img.get_fdata(dtype=np.float32), weights, axes=([3], [0]))
            weighting = (
                f'weighted by the number of DWI volumes each describes ({index_key})'
                if index_key in sidecar
                else 'with equal weights'
            )
            steps.append(f'The {img.shape[3]} volumes were averaged, {weighting}')
        elif img.ndim == 4:
            data = img.get_fdata(dtype=np.float32)[..., 0]

        scalar_img = img if data is None else nb.Nifti1Image(data, img.affine)
        scalar_img, resampled = self._on_grid(scalar_img)
        steps += resampled
        if not steps and data is None:
            return self._scalar(variable_name, qc_file, qc_file, sidecar, steps, extra=extra)
        return self._scalar(
            variable_name, scalar_img, qc_file, sidecar, steps, write=True, extra=extra
        )

    def _scalar(
        self,
        variable_name,
        img,
        qc_file,
        sidecar,
        steps,
        write=False,
        definition=None,
        desc=None,
        extra=None,
    ):
        """Assemble (variable name, definition, path, run-specific metadata).

        ``img`` is written to the working directory if ``write`` is set,
        otherwise it is the path of a QSIPrep map used as it is.
        """
        if write:
            path = op.join(self._cwd, f'{variable_name}.nii.gz')
            img.to_filename(path)
        else:
            path = img
        metadata = {
            # Keep what QSIPrep says about the map, without the per-volume indices
            'QSIPrepMetadata': {
                key: value for key, value in sidecar.items() if not key.endswith('Index')
            },
            'QSIPrepFile': op.basename(qc_file),
        }
        if steps:
            metadata['QSIReconProcessing'] = steps
        metadata.update(extra or {})
        definition = deepcopy(self.scalar_metadata[definition or variable_name])
        if desc:
            definition['bids']['desc'] = desc
        return variable_name, definition, path, metadata

    def _on_grid(self, img, order=1):
        """Resample a 3D image onto the DWI grid if it is not already there."""
        shape, affine = self._grid
        if img.shape[:3] == tuple(shape) and np.allclose(img.affine, affine, atol=1e-3):
            return img, []
        resampled = resample_from_to(img, (shape, affine), order=order)
        kind = 'linearly' if order == 1 else 'with nearest-neighbor interpolation'
        return resampled, [f'Resampled {kind} onto the grid of the preprocessed DWI']


def _load_sidecar(in_file):
    sidecar = re.sub(r'\.nii(\.gz)?$', '.json', in_file)
    if not op.exists(sidecar):
        return {}
    with open(sidecar) as fobj:
        return json.load(fobj)


def _volume_weights(index, n_maps):
    """Weight each map volume by the number of DWI volumes it describes.

    ``index`` gives, for each DWI volume, the map volume that describes it.
    Without a usable index, every map volume gets the same weight.
    """
    weights = np.ones(n_maps)
    if index:
        index = np.asarray(index, dtype=int)
        if index.min() >= 0 and index.max() < n_maps:
            weights = np.bincount(index, minlength=n_maps).astype(float)
    return weights / weights.sum()


def _shells(bval_file):
    """Return the rounded b-values of the nonzero shells, in ascending order."""
    from .gradients import _cluster_bvals

    bvals = np.atleast_1d(np.loadtxt(bval_file))
    bvals = bvals[bvals > B0_THRESHOLD]
    if bvals.size == 0:
        return []
    if bvals.size == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = _cluster_bvals(bvals, SHELL_DISTANCE)
    medians = sorted(np.median(bvals[labels == label]) for label in np.unique(labels))
    return [int(round(median / 10) * 10) for median in medians]
