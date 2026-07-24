"""Utility functions for tests."""

import base64
import lzma
import os
import shutil
import tarfile
from glob import glob
from gzip import GzipFile
from io import BytesIO

import nibabel as nb
import numpy as np
import requests
from nipype import logging

from qsirecon import config

LOGGER = logging.getLogger('nipype.utils')


def download_test_data(dset, data_dir=None):
    """Download test data."""
    URLS = {
        'multishell_output': (
            'https://upenn.box.com/shared/static/q066t33ojqk1phljr6064ilcrrqip87q.gz'
        ),
        'singleshell_output': (
            'https://upenn.box.com/shared/static/rc4oh9tqf20f2xu2p8587q44vwr1i7uf.gz'
        ),
        'hsvs_data': ('https://upenn.box.com/shared/static/kic35jscwqhtezi6t398k4nfuo09ppty.xz'),
        'multises_pre1_output': (
            'https://upenn.box.com/shared/static/g6hb4ylraejqn8sjz2xodj6fog4lp6sb.xz'
        ),
        'multises_post1_output': (
            'https://upenn.box.com/shared/static/ipqhy6a9p0pl7q1tw4zejj47mro4dtfh.xz'
        ),
        'custom_atlases': (
            'https://upenn.box.com/shared/static/2x5jqj7he20lminc3v4jtwhyiq7pyr8i.gz'
        ),
    }
    if dset == '*':
        for k in URLS:
            download_test_data(k, data_dir=data_dir)

        return

    if dset not in URLS:
        raise ValueError(f'dset ({dset}) must be one of: {", ".join(URLS.keys())}')

    if not data_dir:
        data_dir = os.path.join(os.path.dirname(get_test_data_path()), 'test_data')

    out_dir = os.path.join(data_dir, dset)

    if os.path.isdir(out_dir):
        config.loggers.utils.info(
            f'Dataset {dset} already exists. '
            'If you need to re-download the data, please delete the folder.'
        )
        return out_dir
    else:
        config.loggers.utils.info(f'Downloading {dset} to {out_dir}')

    os.makedirs(out_dir, exist_ok=True)
    url = URLS[dset]
    with requests.get(url, stream=True) as req:
        if url.endswith('.xz'):
            with lzma.open(BytesIO(req.content)) as f:
                with tarfile.open(fileobj=f) as t:
                    t.extractall(out_dir)
        elif url.endswith('.gz'):
            with tarfile.open(fileobj=GzipFile(fileobj=BytesIO(req.content))) as t:
                t.extractall(out_dir)
        else:
            raise ValueError(f'Unknown file type for {dset} ({url})')

    return out_dir


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
