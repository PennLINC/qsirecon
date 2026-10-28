"""Tests for the Restriction Spectrum Imaging (RSI) port and interface."""

import importlib.util
import re
import shutil
import subprocess
import sys

import nibabel as nb
import numpy as np
import pytest

from qsirecon.data import load as load_data
from qsirecon.interfaces import rsi
from qsirecon.interfaces.recon_scalars import RSIReconScalars, rsi_scalars
from qsirecon.utils import rsi as rsi_utils

CUSTOM_SETTINGS = {
    'restricted_parallel_diffusivity': 1.2e-3,
    'restricted_perpendicular_diffusivity': 1.0e-5,
    'restricted_sh_degree': 6,
    'hindered_parallel_diffusivity': 1.1e-3,
    'hindered_perpendicular_diffusivity': 0.7e-3,
    'hindered_sh_degree': 2,
    'free_diffusivity': 2.5e-3,
    'regularization_lambda': 0.25,
}


def _write_rsi_inputs(tmpdir, b0_value=0, shape=(3, 3, 3)):
    """Write a noisy multi-shell DWI series with a mix of compartments in each voxel.

    Voxels with i=0 are mostly a stick along x and voxels with i=1 are mostly free water.
    """
    rng = np.random.default_rng(42)
    bvecs = rng.normal(size=(45, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = np.repeat([1000.0, 2000.0, 3000.0], 15)
    bvecs = np.vstack([np.zeros((3, 3)), bvecs])
    bvals = np.concatenate([[b0_value] * 3, bvals])

    stick = np.exp(-bvals * 1.0e-3 * bvecs[:, 0] ** 2)
    hindered = np.exp(-bvals * (0.9e-3 + 0.1e-3 * bvecs[:, 1] ** 2))
    free_water = np.exp(-bvals * 3.0e-3)
    fractions = rng.dirichlet([1, 1, 1], size=shape)
    fractions[0] = [0.9, 0.05, 0.05]
    fractions[1] = [0.05, 0.05, 0.9]
    signal = fractions @ np.vstack([stick, hindered, free_water])
    data = 1000 * signal + rng.normal(scale=5, size=signal.shape)

    mask = np.ones(shape, dtype=np.uint8)
    mask[-1, -1, -1] = 0

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    paths = {
        'dwi_file': tmpdir / 'sub-01_dwi.nii.gz',
        'mask_file': tmpdir / 'sub-01_mask.nii.gz',
        'bval_file': tmpdir / 'sub-01_dwi.bval',
        'bvec_file': tmpdir / 'sub-01_dwi.bvec',
    }
    nb.Nifti1Image(data.astype(np.float32), affine).to_filename(paths['dwi_file'])
    nb.Nifti1Image(mask, affine).to_filename(paths['mask_file'])
    np.savetxt(paths['bval_file'], bvals[None, :], fmt='%g')
    np.savetxt(paths['bvec_file'], bvecs.T, fmt='%.8f')
    return {key: str(value) for key, value in paths.items()}


def _upstream_constants(settings):
    """Format interface settings as the constant definitions in RSIproc."""
    return {
        'RSI_lambda': f'{settings["regularization_lambda"]!r}',
        'RSI_ADC_long': 'np.array([{!r}, {!r}, {!r}])'.format(
            settings['restricted_parallel_diffusivity'],
            settings['hindered_parallel_diffusivity'],
            settings['free_diffusivity'],
        ),
        'RSI_ADC_trans': 'np.array([{!r}, {!r}, {!r}])'.format(
            settings['restricted_perpendicular_diffusivity'],
            settings['hindered_perpendicular_diffusivity'],
            settings['free_diffusivity'],
        ),
        'RSI_SH_order': 'np.array([{}, {}, 0])'.format(
            settings['restricted_sh_degree'], settings['hindered_sh_degree']
        ),
    }


def _run_upstream(inputs, out_dir, constants=None):
    """Run the vendored RSIproc script, optionally with its constants replaced."""
    script_dir = out_dir / 'script'
    script_dir.mkdir(parents=True)
    script = load_data(rsi_utils.RSI_SCRIPT).read_text()
    for name, value in (constants or {}).items():
        script, n_subs = re.subn(
            rf'^{name}\s*=.*$', f'{name} = {value}', script, flags=re.MULTILINE
        )
        assert n_subs == 1, f'Could not replace {name} in {rsi_utils.RSI_SCRIPT}'

    script_file = script_dir / 'RSIproc.py'
    script_file.write_text(script)
    shutil.copy(load_data('rsi/icosahedron.py'), script_dir / 'icosahedron.py')
    subprocess.run(
        [
            sys.executable,
            str(script_file),
            inputs['dwi_file'],
            inputs['mask_file'],
            inputs['bval_file'],
            inputs['bvec_file'],
        ],
        cwd=out_dir,
        check=True,
    )
    return {
        upstream_name: nb.load(out_dir / f'{upstream_name}.nii.gz').get_fdata()
        for _, _, upstream_name in rsi.RSI_OUTPUTS.values()
    }


def _assert_matches_upstream(results, upstream):
    for output_name, (_, _, upstream_name) in rsi.RSI_OUTPUTS.items():
        ported = nb.load(getattr(results.outputs, output_name)).get_fdata()
        np.testing.assert_allclose(
            ported,
            upstream[upstream_name],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f'{output_name} does not match RSIproc {upstream_name}',
        )


def test_rsi_constants_match_vendored_script():
    """The default settings must match the values hard-coded in RSIproc."""
    script = load_data(rsi_utils.RSI_SCRIPT).read_text()

    def _get_constant(name):
        match = re.search(rf'^{name}\s*=\s*(.+)$', script, flags=re.MULTILINE)
        assert match, f'{name} not found in {rsi_utils.RSI_SCRIPT}'
        value = match.group(1)
        if value.startswith('np.array('):
            return [float(v) for v in re.findall(r'[-+.\deE]+', value[len('np.array(') :])]
        return float(value)

    assert _get_constant('RSI_lambda') == rsi_utils.RSI_LAMBDA
    assert _get_constant('RSI_ADC_long') == rsi_utils.RSI_ADC_LONG
    assert _get_constant('RSI_ADC_trans') == rsi_utils.RSI_ADC_TRANS
    assert _get_constant('RSI_SH_order') == rsi_utils.RSI_SH_ORDER
    assert 'make_icosahedron(3)' in script


def test_make_icosahedron_matches_upstream():
    """The ported mesh has the same vertices as RSIproc's icosahedron module."""
    spec = importlib.util.spec_from_file_location(
        'rsi_icosahedron', load_data('rsi/icosahedron.py')
    )
    icosahedron = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(icosahedron)

    upstream = icosahedron.make_icosahedron(3)
    ported = rsi_utils.make_icosahedron(3)
    assert ported.shape == upstream.shape == (642, 3)
    distances = np.linalg.norm(upstream[:, None] - ported[None], axis=2)
    assert np.all(distances.min(axis=1) < 1e-12)
    assert len(set(distances.argmin(axis=1))) == len(upstream)


def test_rsi_matches_upstream(tmp_path):
    """With default settings, the port reproduces every RSIproc output."""
    inputs = _write_rsi_inputs(tmp_path)
    upstream = _run_upstream(inputs, tmp_path / 'upstream')

    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    results = rsi.RSI(**inputs).run(cwd=str(work_dir))
    _assert_matches_upstream(results, upstream)


def test_rsi_matches_upstream_custom_settings(tmp_path):
    """With custom settings, the port matches RSIproc edited to use the same settings."""
    inputs = _write_rsi_inputs(tmp_path)
    upstream = _run_upstream(
        inputs, tmp_path / 'upstream', constants=_upstream_constants(CUSTOM_SETTINGS)
    )

    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    results = rsi.RSI(**inputs, **CUSTOM_SETTINGS).run(cwd=str(work_dir))
    _assert_matches_upstream(results, upstream)

    metadata = results.outputs.restrictediso_metadata
    assert metadata['Model']['Parameters']['Lambda'] == 0.25
    assert metadata['Model']['Parameters']['Compartments'] == {
        'restricted': {
            'ParallelDiffusivity': 1.2e-3,
            'PerpendicularDiffusivity': 1.0e-5,
            'SphericalHarmonicsDegree': 6,
        },
        'hindered': {
            'ParallelDiffusivity': 1.1e-3,
            'PerpendicularDiffusivity': 0.7e-3,
            'SphericalHarmonicsDegree': 2,
        },
        'free': {'IsotropicDiffusivity': 2.5e-3},
    }


@pytest.mark.parametrize('b0_value', [0, 30])
def test_rsi(tmp_path, b0_value):
    """RSI writes all scalar maps and separates restricted signal from free water."""
    inputs = _write_rsi_inputs(tmp_path, b0_value=b0_value)
    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    results = rsi.RSI(**inputs).run(cwd=str(work_dir))

    for output_name in rsi.RSI_OUTPUTS:
        img = nb.load(getattr(results.outputs, output_name))
        assert img.shape == (3, 3, 3)
        assert img.get_data_dtype() == np.float32
        assert np.all(np.isfinite(img.get_fdata()))
        metadata = getattr(results.outputs, f'{output_name}_metadata')
        assert metadata['Software']['Version'] == rsi_utils.RSI_VERSION
        assert metadata['Model']['Parameters']['b0_threshold'] == 50.0
        assert metadata['Model']['Parameters']['Lambda'] == rsi_utils.RSI_LAMBDA
        assert metadata['Model']['Parameters']['Compartments'] == {
            'restricted': {
                'ParallelDiffusivity': 1.0e-3,
                'PerpendicularDiffusivity': 1.0e-10,
                'SphericalHarmonicsDegree': 4,
            },
            'hindered': {
                'ParallelDiffusivity': 1.0e-3,
                'PerpendicularDiffusivity': 0.9e-3,
                'SphericalHarmonicsDegree': 4,
            },
            'free': {'IsotropicDiffusivity': 3.0e-3},
        }

    restricted_iso = nb.load(results.outputs.restrictediso).get_fdata()
    restricted_dir = nb.load(results.outputs.restricteddir).get_fdata()
    free_iso = nb.load(results.outputs.freeiso).get_fdata()
    # Stick voxels are mostly restricted and free water voxels are mostly free water
    assert np.all(restricted_dir[0] > restricted_dir[1])
    assert np.all(restricted_iso[0] > free_iso[0])
    assert np.all(free_iso[1] > restricted_iso[1])
    assert np.all(free_iso[1] > 0.5)
    # Voxels outside the mask are not fit
    assert restricted_iso[-1, -1, -1] == 0


def test_rsi_requires_b0(tmp_path):
    """RSI fails with a clear error when there are no b=0 images."""
    inputs = _write_rsi_inputs(tmp_path, b0_value=500)
    with pytest.raises(ValueError, match='at least one b=0'):
        rsi.RSI(**inputs).run(cwd=str(tmp_path))


def test_rsi_rejects_odd_sh_degree():
    """Spherical harmonic degrees must be even and positive."""
    with pytest.raises(Exception, match='restricted_sh_degree'):
        rsi.RSI(restricted_sh_degree=3)


def test_describe_rsi_model():
    """The boilerplate describes the settings actually used."""
    defaults = rsi.describe_rsi_model(rsi.RSI().inputs)
    assert 'parallel diffusivity 0.001 mm^2/s' in defaults
    assert 'perpendicular diffusivity 1e-10 mm^2/s' in defaults
    assert 'perpendicular diffusivity 0.0009 mm^2/s' in defaults
    assert 'isotropic diffusivity 0.003 mm^2/s' in defaults
    assert 'spherical harmonics up to degree 4, and' in defaults
    assert 'lambda = 0.1)' in defaults

    custom = rsi.describe_rsi_model(rsi.RSI(**CUSTOM_SETTINGS).inputs)
    assert 'parallel diffusivity 0.0012 mm^2/s' in custom
    assert 'isotropic diffusivity 0.0025 mm^2/s' in custom
    assert 'up to degree 6 and 2, respectively' in custom
    assert 'lambda = 0.25)' in custom


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
