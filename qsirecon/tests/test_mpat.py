"""Tests for the MPAT recon workflow."""

from copy import deepcopy

import pytest
import yaml

from qsirecon import config
from qsirecon.data import load as load_data
from qsirecon.workflows.recon.mpat import _mpat_memory_budget, init_mpat_wf


def _builtin_spec():
    return yaml.safe_load(load_data.readable('pipelines/mpat_tractography.yaml').read_text())


@pytest.fixture
def workflow_warnings(monkeypatch):
    """Collect the arguments of every warning logged on the workflow logger."""
    warnings = []
    monkeypatch.setattr(config.loggers.workflow, 'warning', lambda *args: warnings.append(args))
    return warnings


@pytest.fixture
def nipype_config(monkeypatch, tmp_path):
    """Give the workflow builder thread, memory and output settings, as the CLI would."""
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 4)
    monkeypatch.setattr(config.nipype, 'memory_gb', None)
    monkeypatch.setattr(config.execution, 'output_dir', tmp_path)
    return config.nipype


@pytest.mark.parametrize(
    ('memory_gb', 'expected'),
    [
        (None, 'auto'),
        (0, 'auto'),
        # Half of the memory is the floor
        (16, '8.00GB'),
        # 10 GB of headroom
        (40, '30.00GB'),
        # 80% of the memory
        (64, '51.20GB'),
    ],
)
def test_mpat_memory_budget(memory_gb, expected):
    """MPAT's memory budget follows --mem as MPAT's own auto setting follows the job's limit."""
    assert _mpat_memory_budget(memory_gb) == expected


def test_init_mpat_wf_requires_anatomical_data():
    """Workflow construction fails early when the preprocessed data have no anatomical image."""
    with pytest.raises(ValueError, match='requires an anatomical image'):
        init_mpat_wf({'has_qsiprep_anat': False}, name='mpat', params={})


@pytest.mark.parametrize(
    ('params', 'match'),
    [
        ({'angle': 35}, 'Unknown MPAT parameters: angle'),
        ({'mpat': {'n_seeds': 1}}, 'Unknown parameters in the "mpat" section: n_seeds'),
    ],
)
def test_init_mpat_wf_rejects_unknown_parameters(params, match, nipype_config):
    """Parameters MPAT would not use are errors, not silently ignored."""
    with pytest.raises(ValueError, match=match):
        init_mpat_wf({'has_qsiprep_anat': True}, name='mpat', params=params)


def test_builtin_spec_builds(nipype_config, workflow_warnings):
    """The packaged spec builds a workflow whose MPAT node gets the spec's settings."""
    spec = _builtin_spec()
    assert spec['space'] == 'T1w'
    (node,) = spec['nodes']
    assert (node['software'], node['action']) == ('MPAT', 'tractography')

    wf = init_mpat_wf(
        {'has_qsiprep_anat': True},
        name=node['name'],
        qsirecon_suffix=node['qsirecon_suffix'],
        params=deepcopy(node['parameters']),
    )
    # Every pyAFQ setting in the spec is one the installed pyAFQ knows
    assert workflow_warnings == []

    node = wf.get_node('run_mpat')
    inputs = node.inputs
    assert inputs.n_procs == 4
    # Without --mem, MPAT reads the job's memory limit itself,
    # and the node reserves enough of Nipype's memory that no other MPAT node runs with it
    from nipype.utils.profiler import get_system_total_memory_gb

    assert inputs.max_mem == 'auto'
    assert 0.5 * get_system_total_memory_gb() < node.mem_gb < 0.9 * get_system_total_memory_gb()
    assert inputs.angle == 35
    assert inputs.tol == 1e-6
    assert inputs.double_odf is True
    assert inputs.moments is False
    assert inputs.bundles is None
    assert inputs.kwargs['tracking_params'] == {'odf_model': 'CSD'}
    assert inputs.kwargs['segmentation_params'] == {'dist_to_atlas': 4}
    assert inputs.kwargs['bundle_info'] is None
    assert 'ds_mpat_tractography' in wf.list_node_names()
    assert '@mpat' in wf.__desc__


def test_builtin_spec_uses_mem(nipype_config):
    """An "auto" memory budget in the spec follows --mem when it is given."""
    nipype_config.memory_gb = 40
    (node,) = _builtin_spec()['nodes']
    wf = init_mpat_wf({'has_qsiprep_anat': True}, name='mpat', params=node['parameters'])
    node = wf.get_node('run_mpat')
    assert node.inputs.max_mem == '30.00GB'
    assert node.mem_gb == 40


def test_sloppy_spec(nipype_config):
    """Sloppy mode cuts MPAT down to two bundles, a loose tolerance and an affine mapping."""
    from qsirecon.utils.sloppy_recon import make_sloppy

    (node,) = make_sloppy(_builtin_spec())['nodes']
    assert node['parameters']['mpat']['bundles'] == ['Left Arcuate', 'Right Corticospinal']
    assert node['parameters']['mpat']['tol'] == 1e-3
    assert node['parameters']['mpat']['viz'] is False
    assert node['parameters']['pyafq']['mapping_definition'] == 'AffMap()'

    wf = init_mpat_wf({'has_qsiprep_anat': True}, name='mpat', params=node['parameters'])
    inputs = wf.get_node('run_mpat').inputs
    assert inputs.bundles == ['Left Arcuate', 'Right Corticospinal']
    assert type(inputs.kwargs['mapping_definition']).__name__ == 'AffMap'


def test_build_workflow_dispatches_to_mpat(nipype_config):
    """Recon specs reach the MPAT workflow through software "MPAT" and action "tractography"."""
    from qsirecon.workflows.recon.build_workflow import workflow_from_spec

    (node,) = _builtin_spec()['nodes']
    wf = workflow_from_spec({'has_qsiprep_anat': True}, node)
    assert wf.name == 'mpat_tractography'
    assert 'run_mpat' in wf.list_node_names()


def test_default_bundle_dict():
    """By default MPAT computes pyAFQ's bundles, without the ones that need SynthSeg."""
    from qsirecon.interfaces.mpat import default_bundle_dict

    names = list(default_bundle_dict())
    for name in (
        'Left Arcuate',
        'Right Corticospinal',
        'Left Anterior Thalamic',
        'Left Superior Longitudinal I',
        'Callosum Motor',
    ):
        assert name in names

    assert not any('Fornix' in name or 'Cingulum Hippocampus' in name for name in names)


def test_mpat_interface_requires_pve(tmp_path):
    """The interface refuses to run MPAT without tissue probability maps."""
    import nibabel as nib
    import numpy as np

    from qsirecon.interfaces.mpat import MPATRecon

    img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    files = {}
    for name in ('dwi_file', 't1_file', 'mask_file'):
        files[name] = str(tmp_path / f'{name}.nii.gz')
        img.to_filename(files[name])

    for name in ('bval_file', 'bvec_file'):
        files[name] = str(tmp_path / name)
        (tmp_path / name).write_text('0\n')

    interface = MPATRecon(kwargs={}, **files)
    interface.inputs.csf_probseg = None
    with pytest.raises(ValueError, match='probability maps'):
        interface.run(cwd=str(tmp_path))
