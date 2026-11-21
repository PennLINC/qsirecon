"""Utility functions for tests."""

import base64
import os
import shutil
from glob import glob

import nibabel as nb
import numpy as np
from nipype import logging

from qsirecon import config
from qsirecon.tests.integration_test_data import download_test_data as _download_test_data

LOGGER = logging.getLogger('nipype.utils')


def download_test_data(dset, data_dir=None):
    """Download test data."""
    return _download_test_data(
        dset,
        data_dir=data_dir,
        info=config.loggers.utils.info,
    )


def shrink_dataset_fov(src_qsiprep_dir, dest_qsiprep_dir, target_fov_mm=100.0):
    """Copy a QSIPrep-derivative tree and uniformly shrink the physical field of view.

    Every ``.nii``/``.nii.gz`` under ``src_qsiprep_dir`` is copied to ``dest_qsiprep_dir``
    and its affine is left-multiplied by ``diag([S, S, S, 1])`` (and its three spatial
    pixdims scaled by ``S``), where ``S`` is chosen so the largest field-of-view axis of the
    preprocessed DWI becomes ``target_fov_mm`` millimeters. The voxel matrix and data are
    unchanged -- this is a header-only edit, so it is near-instant regardless of image size,
    and it needs no gradient (bvec) rotation because a uniform scale has no rotational
    component. The identical scaling is applied to *all* images so they stay mutually
    registered (the recon brain mask is resampled onto the DWI grid with an identity
    transform, so a partial shrink would misalign it). ``.bval``/``.bvec``/``.json``/``.tsv``
    and ``.h5`` transforms are copied verbatim.

    This is used to synthesize a neonate-sized ("small head") input from an adult test
    dataset, so DSI Studio's internal ``is_human_size`` field-of-view check (~130 mm
    threshold) selects a non-adult template -- the condition that triggers the HBCD "Chen"
    autotrack crash on a broken DSI Studio build.

    Parameters
    ----------
    src_qsiprep_dir : str
        Path to a QSIPrep-derivative directory (the qsirecon input directory).
    dest_qsiprep_dir : str
        Where to write the shrunken copy. Overwritten if it already exists.
    target_fov_mm : float
        Target size (mm) for the DWI's largest field-of-view axis. Default 100.0, which is
        neonate-sized and comfortably below DSI Studio's ~130 mm adult threshold.

    Returns
    -------
    str
        ``dest_qsiprep_dir`` (for use as the qsirecon input directory).
    """
    if os.path.isdir(dest_qsiprep_dir):
        shutil.rmtree(dest_qsiprep_dir)
    shutil.copytree(src_qsiprep_dir, dest_qsiprep_dir)

    dwi_files = sorted(
        glob(
            os.path.join(dest_qsiprep_dir, '**', '*_desc-preproc_dwi.nii.gz'),
            recursive=True,
        )
    )
    if not dwi_files:
        raise ValueError(f'No *_desc-preproc_dwi.nii.gz found under {dest_qsiprep_dir}')

    ref_img = nb.load(dwi_files[0])
    fov = np.array(ref_img.shape[:3]) * np.array(ref_img.header.get_zooms()[:3])
    scale = float(target_fov_mm) / float(fov.max())

    scale_affine = np.diag([scale, scale, scale, 1.0])
    for nii_file in sorted(glob(os.path.join(dest_qsiprep_dir, '**', '*.nii*'), recursive=True)):
        img = nb.load(nii_file)
        # Materialize the data so we can safely overwrite the file in place.
        data = np.asanyarray(img.dataobj)
        new_affine = scale_affine @ img.affine
        new_img = nb.Nifti1Image(data, new_affine, img.header)

        # Scale the recorded spatial voxel sizes, leaving any 4th (non-spatial) zoom alone.
        zooms = list(img.header.get_zooms())
        new_img.header.set_zooms([z * scale for z in zooms[:3]] + list(zooms[3:]))

        # Write the shrunken affine into both forms, preserving the original codes.
        _, scode = img.header.get_sform(coded=True)
        _, qcode = img.header.get_qform(coded=True)
        new_img.set_sform(new_affine, code=int(scode) if scode else 1)
        new_img.set_qform(new_affine, code=int(qcode) if qcode else 1)

        new_img.to_filename(nii_file)

    return dest_qsiprep_dir


def add_synthetic_qc_maps(src_qsiprep_dir, dest_qsiprep_dir):
    """Copy a QSIPrep-derivative tree and add synthetic voxelwise QC maps to it.

    TEMPORARY: the integration test datasets predate QSIPrep's QC maps, so without this
    the ``qsiprep_qc_scalars`` node is dropped and never runs in the integration tests.
    Remove this once the datasets are regenerated with a QSIPrep that writes the maps.

    Next to each ``*_desc-preproc_dwi.nii.gz``, this writes the maps
    :func:`~qsirecon.interfaces.qc_maps.find_qsiprep_qc_maps` looks for, with sidecars,
    in the layouts that need reducing to 3D:

    * a 4D noise map with one volume per (pretend) series, and ``NoiseMapIndex``
    * a b=0 tSNR map
    * an eddy-style 4D CNR map: the b=0 tSNR, then one volume per shell
    * a 4D Jacobian map with ``JacobianWeightIndex``
    * a 5D ITK displacement field

    The values are smooth fields scaled to plausible ranges, not real estimates.
    A map that already exists is left alone, so this does nothing to a dataset that
    has real QC maps.

    Parameters
    ----------
    src_qsiprep_dir : str
        Path to a QSIPrep-derivative directory (the qsirecon input directory).
    dest_qsiprep_dir : str
        Where to write the copy. Overwritten if it already exists.

    Returns
    -------
    str
        ``dest_qsiprep_dir`` (for use as the qsirecon input directory).
    """
    import json

    from qsirecon.interfaces.qc_maps import (
        B0_THRESHOLD,
        PREPROC_SUFFIX,
        _shells,
        find_qsiprep_qc_maps,
    )

    if os.path.isdir(dest_qsiprep_dir):
        shutil.rmtree(dest_qsiprep_dir)
    shutil.copytree(src_qsiprep_dir, dest_qsiprep_dir)

    dwi_files = sorted(
        glob(os.path.join(dest_qsiprep_dir, '**', f'*_{PREPROC_SUFFIX}.nii.gz'), recursive=True)
    )
    if not dwi_files:
        raise ValueError(f'No *_{PREPROC_SUFFIX}.nii.gz found under {dest_qsiprep_dir}')

    def _write(path, data, affine, sidecar):
        nb.Nifti1Image(data.astype(np.float32), affine).to_filename(path)
        sidecar = {'Description': 'SYNTHETIC map written for the integration tests', **sidecar}
        with open(path.replace('.nii.gz', '.json'), 'w') as fobj:
            json.dump(sidecar, fobj, indent=2)

    for dwi_file in dwi_files:
        existing = find_qsiprep_qc_maps(dwi_file)
        prefix = dwi_file[: -len(f'{PREPROC_SUFFIX}.nii.gz')]
        bval_file = dwi_file.replace('.nii.gz', '.bval')
        bvals = np.atleast_1d(np.loadtxt(bval_file))
        n_vols = bvals.size

        img = nb.load(dwi_file)
        affine = img.affine
        data = img.get_fdata(dtype=np.float32)
        b0 = data[..., bvals <= B0_THRESHOLD].mean(axis=3)
        in_brain = b0 > 0.2 * np.percentile(b0, 99)

        # A smooth field rising from 1 at the center of the grid to 2 at the corners
        grid = np.indices(b0.shape, dtype=np.float32)
        center = (np.array(b0.shape, dtype=np.float32) - 1) / 2
        radius = np.sqrt(
            sum(((grid[axis] - center[axis]) / max(center[axis], 1)) ** 2 for axis in range(3))
        )
        field = 1 + radius / radius.max()

        sigma = 0.02 * np.median(b0[in_brain]) * field
        # Two pretend series, splitting the volumes in half; the second is noisier
        series_index = [0 if vol < n_vols // 2 else 1 for vol in range(n_vols)]
        if 'noise_map_file' not in existing:
            _write(
                f'{prefix}stat-std_desc-noise_dwimap.nii.gz',
                np.stack([sigma, 1.25 * sigma], axis=-1),
                affine,
                {
                    'Units': 'arbitrary',
                    'DenoisingMethod': 'dwidenoise',
                    'Quantity': 'sigma',
                    'NoiseMapIndex': series_index,
                },
            )

        tsnr = np.where(in_brain, b0 / sigma, 0)
        if 'tsnr_file' not in existing:
            _write(f'{prefix}stat-tsnr_dwimap.nii.gz', tsnr, affine, {})

        if 'cnr_file' not in existing:
            shell_cnr = [tsnr * np.exp(-bval * 0.0007) for bval in _shells(bval_file)]
            _write(
                f'{prefix}model-eddy_stat-cnr_dwimap.nii.gz',
                np.stack([tsnr, *shell_cnr], axis=-1),
                affine,
                {},
            )

        # Opposite phase-encoding directions stretch and compress the same region
        stretch = 0.2 * np.exp(-(((grid[1] - center[1]) / max(center[1], 1)) ** 2) * 4)
        if 'jacobian_file' not in existing:
            _write(
                f'{prefix}desc-jacobian_dwimap.nii.gz',
                np.stack([1 + stretch, 1 / (1 + stretch)], axis=-1),
                affine,
                {'SignalRedistributionMethod': 'Jacobian', 'JacobianWeightIndex': series_index},
            )

        if 'sdc_displacement_file' not in existing:
            displacement = np.zeros(b0.shape + (1, 3), dtype=np.float32)
            # Up to 10 mm along the second (phase-encoding) axis, largest at the front
            displacement[..., 0, 1] = 10 * stretch * (grid[1] / max(b0.shape[1] - 1, 1))
            _write(
                f'{prefix}desc-sdc_displacement.nii.gz',
                displacement,
                affine,
                {'EstimationMethod': 'synthetic'},
            )

    return dest_qsiprep_dir


def get_test_data_path():
    """Return the path to test datasets, terminated with separator.

    Test-related data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data') + os.path.sep)


def check_generated_files(output_dir, output_list_file, optional_output_list_file):
    """Compare files generated by qsirecon with a list of expected files."""
    found_files = sorted(glob(os.path.join(output_dir, '**/*'), recursive=True))
    found_files = [os.path.relpath(f, output_dir) for f in found_files]

    # Ignore figures
    found_files = sorted({f for f in found_files if 'figures' not in f})

    # Ignore logs
    found_files = sorted({f for f in found_files if 'log' not in f.split(os.path.sep)})

    with open(output_list_file) as fo:
        expected_files = fo.readlines()
        expected_files = [f.rstrip() for f in expected_files]

    optional_files = []
    if optional_output_list_file:
        with open(optional_output_list_file) as fo:
            optional_files = fo.readlines()
            optional_files = [f.rstrip() for f in optional_files]

    if sorted(found_files) != sorted(expected_files):
        expected_not_found = sorted(set(expected_files) - set(found_files))
        found_not_expected = sorted(set(found_files) - set(expected_files))

        msg = ''
        if expected_not_found:
            msg += '\nExpected but not found:\n\t'
            msg += '\n\t'.join(expected_not_found)

        if found_not_expected:
            # Check that the found files are in the optional file list
            found_not_expected = [f for f in found_not_expected if f not in optional_files]

        if found_not_expected:
            msg += '\nFound but not expected:\n\t'
            msg += '\n\t'.join(found_not_expected)

        if msg:
            raise ValueError(msg)


def reorder_expected_outputs():
    """Load each of the expected output files and sort the lines alphabetically.

    This function is called manually by devs when they modify the test outputs.
    """
    test_data_path = get_test_data_path()
    expected_output_files = sorted(glob(os.path.join(test_data_path, '*_outputs.txt')))
    for expected_output_file in expected_output_files:
        LOGGER.info(f'Sorting {expected_output_file}')

        with open(expected_output_file) as fo:
            file_contents = fo.readlines()

        file_contents = sorted(set(file_contents))

        with open(expected_output_file, 'w') as fo:
            fo.writelines(file_contents)


def freesurfer_license(base_dir):
    """Create a freesurfer license in base_dir."""
    _lic = b'bWFyay5iZXJnbWFuQHVwaHMudXBlbm4uZWR1CjIwMjU5CiAqQ3J2L0QwTXpxcVE2CiBGU3pTdGZYUmlGN2RN'
    license_file = base_dir / 'license.txt'
    with license_file.open('w') as licensef:
        licensef.write(base64.b64decode(_lic).decode('utf-8'))
    return license_file


def get_nodes(wf_results):
    """Load nodes from a Nipype workflow's results."""
    return {node.fullname: node for node in wf_results.nodes}
