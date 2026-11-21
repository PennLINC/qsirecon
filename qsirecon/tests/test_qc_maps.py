"""Tests for using QSIPrep's voxelwise QC maps as scalars."""

import json

import nibabel as nb
import numpy as np
import pytest

from qsirecon.interfaces.ingress import QSIPrepDWIIngress
from qsirecon.interfaces.qc_maps import (
    QSIPrepQCReconScalars,
    find_qsiprep_qc_maps,
    qsiprep_qc_scalars,
    usable_qc_maps,
)
from qsirecon.interfaces.recon_scalars import _parcellation_label
from qsirecon.workflows.recon.build_workflow import _drop_unavailable_qc_nodes

PREFIX = 'sub-01_space-ACPC_res-2mm_'
SHAPE = (4, 4, 3)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
SCALAR_KEYS = {'bids', 'metadata', 'path', 'qsirecon_suffix', 'variable_name', 'source_file'}


def _write(path, data, affine=AFFINE, sidecar=None):
    nb.Nifti1Image(np.asarray(data, dtype=np.float32), affine).to_filename(path)
    if sidecar is not None:
        path.with_name(path.name.replace('.nii.gz', '.json')).write_text(json.dumps(sidecar))
    return str(path)


@pytest.fixture
def dwi_dir(tmp_path):
    """A QSIPrep-style dwi directory holding a preprocessed series and its gradients."""
    out_dir = tmp_path / 'sub-01' / 'dwi'
    out_dir.mkdir(parents=True)
    _write(out_dir / f'{PREFIX}desc-preproc_dwi.nii.gz', np.ones(SHAPE + (5,)))
    (out_dir / f'{PREFIX}desc-preproc_dwi.bval').write_text('0 1000 1005 2000 1995\n')
    (out_dir / f'{PREFIX}desc-preproc_dwi.bvec').write_text('')
    return out_dir


def _source(dwi_dir):
    return str(dwi_dir / f'{PREFIX}desc-preproc_dwi.nii.gz')


def _run(dwi_dir, **inputs):
    inputs.setdefault('bval_file', str(dwi_dir / f'{PREFIX}desc-preproc_dwi.bval'))
    result = QSIPrepQCReconScalars(
        source_file=_source(dwi_dir),
        qsirecon_suffix='QSIPrepQC',
        dismiss_entities=['desc'],
        **inputs,
    ).run(cwd=str(dwi_dir.parent))
    return {info['variable_name']: info for info in result.outputs.scalar_info}


def test_find_qc_maps_matches_only_this_output(dwi_dir):
    """Maps of another resolution and the native (spaceless) noise map are not picked up."""
    expected = {
        'noise_map_file': f'{PREFIX}stat-std_desc-noise_dwimap.nii.gz',
        'tsnr_file': f'{PREFIX}stat-tsnr_dwimap.nii.gz',
        'cnr_file': f'{PREFIX}model-eddy_stat-cnr_dwimap.nii.gz',
        'jacobian_file': f'{PREFIX}desc-jacobian_dwimap.nii.gz',
        'sdc_displacement_file': f'{PREFIX}desc-sdc_displacement.nii.gz',
    }
    for name in expected.values():
        (dwi_dir / name).write_text('')
    (dwi_dir / 'sub-01_space-ACPC_res-1mm_stat-tsnr_dwimap.nii.gz').write_text('')
    (dwi_dir / 'sub-01_stat-std_desc-noise_dwimap.nii.gz').write_text('')

    found = find_qsiprep_qc_maps(_source(dwi_dir))
    assert {key: value.split('/')[-1] for key, value in found.items()} == expected

    result = QSIPrepDWIIngress(dwi_file=_source(dwi_dir)).run()
    for name, fname in expected.items():
        assert getattr(result.outputs, name).endswith(fname)


def test_find_qc_maps_needs_preproc_name(tmp_path):
    assert find_qsiprep_qc_maps(str(tmp_path / 'sub-01_space-T1w_dwi.nii.gz')) == {}


def test_pooled_noise_and_snr(dwi_dir):
    """A 4D noise map is pooled over series, weighted by the volumes each describes."""
    noise = np.stack([np.full(SHAPE, 1.0), np.full(SHAPE, 2.0)], axis=-1)
    noise_file = _write(
        dwi_dir / f'{PREFIX}stat-std_desc-noise_dwimap.nii.gz',
        noise,
        sidecar={'Quantity': 'sigma', 'NoiseMapIndex': [0, 0, 0, 1, 0]},
    )
    dwi_ref = _write(dwi_dir / f'{PREFIX}desc-preproc_dwiref.nii.gz', np.full(SHAPE, 100.0))
    mask = np.ones(SHAPE)
    mask[0] = 0
    dwi_mask = _write(dwi_dir / 'mask.nii.gz', mask)

    scalars = _run(dwi_dir, noise_map_file=noise_file, dwi_ref=dwi_ref, dwi_mask=dwi_mask)
    assert set(scalars) == {'pooledstd_noise', 'snr'}

    sigma = np.sqrt((4 * 1.0 + 1 * 4.0) / 5)
    noise_img = nb.load(scalars['pooledstd_noise']['path'])
    assert noise_img.shape == SHAPE
    np.testing.assert_allclose(noise_img.get_fdata(), sigma, rtol=1e-6)
    assert 'NoiseMapIndex' not in scalars['pooledstd_noise']['metadata']['QSIPrepMetadata']
    assert scalars['pooledstd_noise']['bids'] == {'stat': 'pooledstd', 'desc': 'noise'}
    assert _parcellation_label(scalars['pooledstd_noise']) == 'pooledstd_noise'

    snr = nb.load(scalars['snr']['path']).get_fdata()
    np.testing.assert_allclose(snr[1:], 100.0 / sigma, rtol=1e-5)
    assert np.all(snr[0] == 0)

    for info in scalars.values():
        assert SCALAR_KEYS <= set(info)
        assert info['qsirecon_suffix'] == 'QSIPrepQC'


def test_residual_rms_has_no_snr(dwi_dir):
    noise_file = _write(
        dwi_dir / f'{PREFIX}stat-rms_desc-noise_dwimap.nii.gz',
        np.ones(SHAPE),
        sidecar={'Quantity': 'residual_rms'},
    )
    dwi_ref = _write(dwi_dir / f'{PREFIX}desc-preproc_dwiref.nii.gz', np.ones(SHAPE))
    scalars = _run(dwi_dir, noise_map_file=noise_file, dwi_ref=dwi_ref)
    assert set(scalars) == {'pooledrms_noise'}
    # A 3D map on the DWI grid is used as it is
    assert scalars['pooledrms_noise']['path'] == noise_file


def test_eddy_cnr_is_split_by_shell(dwi_dir):
    cnr = np.stack([np.full(SHAPE, value) for value in (7.0, 3.0, 2.0)], axis=-1)
    cnr_file = _write(dwi_dir / f'{PREFIX}model-eddy_stat-cnr_dwimap.nii.gz', cnr, sidecar={})
    scalars = _run(dwi_dir, cnr_file=cnr_file)

    assert set(scalars) == {'cnr_b1000', 'cnr_b2000'}
    assert scalars['cnr_b1000']['bids'] == {'model': 'eddy', 'stat': 'cnr', 'desc': 'b1000'}
    assert np.all(nb.load(scalars['cnr_b1000']['path']).get_fdata() == 3.0)
    assert np.all(nb.load(scalars['cnr_b2000']['path']).get_fdata() == 2.0)
    assert 'eddy model' in scalars['cnr_b2000']['metadata']['Description']
    assert _parcellation_label(scalars['cnr_b2000']) == 'cnr_b2000'
    # The definitions loaded from YAML must not be changed
    assert 'desc' not in qsiprep_qc_scalars['cnr']['bids']


def test_cnr_with_unexpected_volumes_is_skipped(dwi_dir):
    cnr_file = _write(
        dwi_dir / f'{PREFIX}model-eddy_stat-cnr_dwimap.nii.gz', np.ones(SHAPE + (5,))
    )
    tsnr_file = _write(dwi_dir / f'{PREFIX}stat-tsnr_dwimap.nii.gz', np.ones(SHAPE))
    assert set(_run(dwi_dir, cnr_file=cnr_file, tsnr_file=tsnr_file)) == {'tsnr'}


def test_jacobian_4d_is_averaged_by_index(dwi_dir):
    jacobian = np.stack([np.full(SHAPE, 1.0), np.full(SHAPE, 3.0)], axis=-1)
    jacobian_file = _write(
        dwi_dir / f'{PREFIX}desc-jacobian_dwimap.nii.gz',
        jacobian,
        sidecar={'JacobianWeightIndex': [0, 1, 1, 1, 0]},
    )
    scalars = _run(dwi_dir, jacobian_file=jacobian_file)
    np.testing.assert_allclose(nb.load(scalars['mean_jacobian']['path']).get_fdata(), 2.2)
    assert 'JacobianWeightIndex' in scalars['mean_jacobian']['metadata']['QSIReconProcessing'][0]


def test_displacement_magnitude(dwi_dir):
    vectors = np.zeros(SHAPE + (1, 3))
    vectors[..., 0, 0] = 3.0
    vectors[..., 0, 1] = -4.0
    displacement_file = _write(dwi_dir / f'{PREFIX}desc-sdc_displacement.nii.gz', vectors)
    scalars = _run(dwi_dir, sdc_displacement_file=displacement_file)
    np.testing.assert_allclose(nb.load(scalars['magnitude_sdc']['path']).get_fdata(), 5.0)
    assert scalars['magnitude_sdc']['metadata']['Units'] == 'mm'


def test_off_grid_map_is_resampled(dwi_dir):
    affine = AFFINE.copy()
    affine[:3, 3] = 0.5
    tsnr_file = _write(dwi_dir / f'{PREFIX}stat-tsnr_dwimap.nii.gz', np.ones((3, 3, 3)), affine)
    scalars = _run(dwi_dir, tsnr_file=tsnr_file)
    img = nb.load(scalars['tsnr']['path'])
    assert img.shape == SHAPE
    np.testing.assert_allclose(img.affine, AFFINE)
    assert scalars['tsnr']['path'] != tsnr_file
    assert 'Resampled' in scalars['tsnr']['metadata']['QSIReconProcessing'][0]


def test_zero_tsnr_and_exclusions(dwi_dir):
    """An all-zero tSNR map is dropped, and maps can be left out with ``include``."""
    tsnr_file = _write(dwi_dir / f'{PREFIX}stat-tsnr_dwimap.nii.gz', np.zeros(SHAPE))
    jacobian_file = _write(dwi_dir / f'{PREFIX}desc-jacobian_dwimap.nii.gz', np.ones(SHAPE))
    qc_files = {'tsnr_file': tsnr_file, 'jacobian_file': jacobian_file}
    assert usable_qc_maps(qc_files, ['tsnr', 'jacobian']) == {'jacobian_file': jacobian_file}
    assert usable_qc_maps(qc_files, ['tsnr']) == {}

    with pytest.raises(RuntimeError, match='None of the QSIPrep QC maps'):
        _run(dwi_dir, tsnr_file=tsnr_file, jacobian_file=jacobian_file, include=['tsnr'])


def _spec():
    return {
        'nodes': [
            {'action': 'qsiprep_qc_scalars', 'name': 'qc_scalars'},
            {'action': 'export', 'name': 'gqi_scalars'},
            {
                'action': 'bundle_map',
                'name': 'bundle_means_qc_scalars',
                'scalars_from': 'qc_scalars',
            },
            {'action': 'bundle_map', 'name': 'bundle_means_gqi', 'scalars_from': 'gqi_scalars'},
            {
                'action': 'template_map',
                'name': 'tm',
                'scalars_from': ['gqi_scalars', 'qc_scalars'],
            },
        ]
    }


def test_qc_nodes_are_dropped_without_maps(dwi_dir):
    spec = _drop_unavailable_qc_nodes(_spec(), _source(dwi_dir))
    assert [node['name'] for node in spec['nodes']] == ['gqi_scalars', 'bundle_means_gqi', 'tm']
    assert spec['nodes'][-1]['scalars_from'] == ['gqi_scalars']


def test_qc_nodes_are_kept_with_maps(dwi_dir):
    _write(dwi_dir / f'{PREFIX}stat-tsnr_dwimap.nii.gz', np.ones(SHAPE))
    spec = _drop_unavailable_qc_nodes(_spec(), _source(dwi_dir))
    assert spec == _spec()


def test_qc_workflow_builds(tmp_path, monkeypatch):
    from qsirecon import config
    from qsirecon.workflows.recon.qc_scalars import init_qsiprep_qc_scalars_wf

    monkeypatch.setattr(config.execution, 'output_dir', tmp_path)
    workflow = init_qsiprep_qc_scalars_wf({}, qsirecon_suffix='QSIPrepQC')
    assert 'recon_scalars' in workflow.list_node_names()

    with pytest.raises(ValueError, match='Unknown QSIPrep QC maps'):
        init_qsiprep_qc_scalars_wf({}, params={'include': ['fieldmap']})


@pytest.mark.parametrize(
    ('entities', 'expected'),
    [
        ({'model': 'eddy', 'stat': 'cnr', 'desc': 'b1000'}, 'model-eddy_stat-cnr_desc-b1000'),
        ({'stat': 'pooledstd', 'desc': 'noise'}, 'stat-pooledstd_desc-noise'),
        ({'stat': 'pooledrms', 'desc': 'noise'}, 'stat-pooledrms_desc-noise'),
    ],
)
def test_stat_entity_is_written(dwi_dir, tmp_path, entities, expected):
    from qsirecon.interfaces.bids import DerivativesDataSink

    in_file = _write(tmp_path / 'scalar.nii.gz', np.ones(SHAPE))
    result = DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        source_file=_source(dwi_dir),
        in_file=in_file,
        dismiss_entities=['desc'],
        datatype='dwi',
        suffix='dwimap',
        extension='nii.gz',
        **entities,
    ).run()
    assert result.outputs.out_file.endswith(f'_{expected}_dwimap.nii.gz')


def test_model_scalar_labels_are_unchanged():
    assert _parcellation_label({'bids': {'model': 'dki', 'param': 'fa'}}) == 'fa'
    assert _parcellation_label({'bids': {'model': 'rsi', 'param': 'rni', 'desc': 'x'}}) == 'rni_x'


def test_organize_scalar_data_without_param(tmp_path):
    """QC scalars have no param, so the MapNode output must still be set for ds_scalar."""
    import nipype.pipeline.engine as pe

    from qsirecon.interfaces.recon_scalars import OrganizeScalarData

    path = _write(tmp_path / 'scalar.nii.gz', np.ones(SHAPE))
    configs = [
        {'path': path, 'bids': {'stat': 'snr'}},
        {'path': path, 'bids': {'stat': 'pooledstd', 'desc': 'noise'}},
    ]
    node = pe.MapNode(
        OrganizeScalarData(),
        iterfield=['scalar_config'],
        name='organize',
        base_dir=str(tmp_path),
    )
    node.inputs.scalar_config = configs
    outputs = node.run().outputs
    assert outputs.param == [None, None]
    assert outputs.model == [None, None]
    assert outputs.stat == ['snr', 'pooledstd']
