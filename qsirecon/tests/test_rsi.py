"""Tests for the ABCD Study RSI interface."""

import re

import nibabel as nb
import numpy as np
import pytest

from qsirecon.data import load as load_data
from qsirecon.interfaces import rsi
from qsirecon.interfaces.recon_scalars import RSIReconScalars, rsi_scalars


def _write_rsi_inputs(tmpdir, b0_value=0):
    """Write a small DWI series with stick voxels and free water voxels."""
    rng = np.random.default_rng(42)
    bvecs = rng.normal(size=(30, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = np.repeat([1000.0, 2000.0, 3000.0], 10)

    bvecs = np.vstack([np.zeros((2, 3)), bvecs])
    bvals = np.concatenate([[b0_value, b0_value], bvals])

    # Signal from sticks along x where i=0 and free water where i=1.
    # RSIproc squeezes its outputs, so the volume has no singleton dimensions.
    stick = np.exp(-bvals * 1.0e-3 * bvecs[:, 0] ** 2)
    free_water = np.exp(-bvals * 3.0e-3)
    data = np.zeros((2, 2, 2, bvals.size), dtype=np.float32)
    data[0] = 1000 * stick
    data[1] = 1000 * free_water

    affine = np.eye(4)
    dwi_file = tmpdir / 'sub-01_dwi.nii.gz'
    mask_file = tmpdir / 'sub-01_mask.nii.gz'
    bval_file = tmpdir / 'sub-01_dwi.bval'
    bvec_file = tmpdir / 'sub-01_dwi.bvec'
    nb.Nifti1Image(data, affine).to_filename(dwi_file)
    nb.Nifti1Image(np.ones(data.shape[:3], dtype=np.uint8), affine).to_filename(mask_file)
    np.savetxt(bval_file, bvals[None, :], fmt='%g')
    np.savetxt(bvec_file, bvecs.T, fmt='%.8f')
    return {
        'dwi_file': str(dwi_file),
        'mask_file': str(mask_file),
        'bval_file': str(bval_file),
        'bvec_file': str(bvec_file),
    }


def test_rsi_constants_match_vendored_script():
    """The metadata constants must match the values hard-coded in RSIproc."""
    script = load_data(rsi.RSI_SCRIPT).read_text()

    def _get_constant(name):
        match = re.search(rf'^{name}\s*=\s*(.+)$', script, flags=re.MULTILINE)
        assert match, f'{name} not found in {rsi.RSI_SCRIPT}'
        value = match.group(1)
        if value.startswith('np.array('):
            return [float(v) for v in re.findall(r'[-+.\deE]+', value[len('np.array(') :])]
        return float(value)

    assert _get_constant('RSI_lambda') == rsi.RSI_LAMBDA
    assert _get_constant('RSI_ADC_long') == rsi.RSI_ADC_LONG
    assert _get_constant('RSI_ADC_trans') == rsi.RSI_ADC_TRANS
    assert _get_constant('RSI_SH_order') == rsi.RSI_SH_ORDER
    assert 'bval<=10' in script.replace(' ', '')


@pytest.mark.parametrize('b0_value', [0, 30])
def test_rsi(tmp_path, b0_value):
    """RSI writes all scalar maps and separates restricted signal from free water."""
    inputs = _write_rsi_inputs(tmp_path, b0_value=b0_value)
    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    results = rsi.RSI(**inputs).run(cwd=str(work_dir))

    for output_name in rsi.RSI_OUTPUTS:
        img = nb.load(getattr(results.outputs, output_name))
        assert img.shape == (2, 2, 2)
        assert np.all(np.isfinite(img.get_fdata()))
        metadata = getattr(results.outputs, f'{output_name}_metadata')
        assert metadata['Software']['Version'] == rsi.RSI_VERSION
        assert metadata['Model']['Parameters']['b0_threshold'] == 50.0

    n0s1 = nb.load(results.outputs.n0s1).get_fdata()
    nds1 = nb.load(results.outputs.nds1).get_fdata()
    n0s3 = nb.load(results.outputs.n0s3).get_fdata()
    # Stick voxels are mostly restricted and free water voxels are mostly free water
    assert np.all(nds1[0] > nds1[1])
    assert np.all(n0s1[0] > n0s3[0])
    assert np.all(n0s3[1] > n0s1[1])
    assert np.all(n0s3[1] > 0.5)

    # Low b-values are zeroed in a copy of the bval file rather than the original
    shimmed_bval = work_dir / 'sub-01_dwi_rsi.bval'
    assert shimmed_bval.exists() == (b0_value > rsi.RSI_B0_MAX)
    assert np.loadtxt(inputs['bval_file'])[0] == b0_value


def test_rsi_requires_b0(tmp_path):
    """RSI fails with a clear error when there are no b=0 images."""
    inputs = _write_rsi_inputs(tmp_path, b0_value=500)
    with pytest.raises(ValueError, match='at least one b=0'):
        rsi.RSI(**inputs).run(cwd=str(tmp_path))


def test_rsi_recon_scalars(tmp_path):
    """Every RSI output has a scalar definition and vice versa."""
    assert set(rsi_scalars) == set(rsi.RSI_OUTPUTS)

    source_file = tmp_path / 'sub-01_space-T1w_desc-preproc_dwi.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(source_file)
    scalar_files = {}
    for output_name in rsi.RSI_OUTPUTS:
        scalar_files[output_name] = str(tmp_path / f'{output_name}.nii.gz')
        nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(
            scalar_files[output_name]
        )

    results = RSIReconScalars(
        source_file=str(source_file),
        qsirecon_suffix='RSI',
        dismiss_entities=['desc'],
        **scalar_files,
    ).run(cwd=str(tmp_path))
    params = sorted(info['bids']['param'] for info in results.outputs.scalar_info)
    assert params == sorted(rsi.RSI_OUTPUTS)
    assert all(info['bids']['model'] == 'rsi' for info in results.outputs.scalar_info)
