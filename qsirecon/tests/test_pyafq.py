"""Tests for the pyAFQ recon workflow helpers."""

import pytest
import yaml

from qsirecon import config
from qsirecon.data import load as load_data
from qsirecon.workflows.recon.pyafq import _parse_qsirecon_params_dict


@pytest.fixture
def workflow_warnings(monkeypatch):
    """Collect the arguments of every warning logged on the workflow logger."""
    warnings = []
    monkeypatch.setattr(config.loggers.workflow, 'warning', lambda *args: warnings.append(args))
    return warnings


def test_parse_qsirecon_params_dict(workflow_warnings):
    """Spec parameters are nested the way ParticipantAFQ expects, and unknown ones are reported."""
    kwargs = _parse_qsirecon_params_dict(
        {
            'b0_threshold': 50,
            'csd_sh_order_max': '',
            'n_seeds': 1,
            'clip_edges': False,
            'distance_threshold': 3,
            'export': 'all',
            'use_external_tracking': False,
            'not_a_pyafq_parameter': 1,
            # positional argument of pyAFQ's track(), which pyAFQ supplies itself
            'params_file': 'x',
        }
    )
    assert kwargs['b0_threshold'] == 50
    # Empty strings mean None
    assert kwargs['csd_sh_order_max'] is None
    assert kwargs['tracking_params'] == {'n_seeds': 1}
    assert kwargs['segmentation_params'] == {
        'clip_edges': False,
        'cleaning_params': {'distance_threshold': 3},
    }
    # QSIRecon consumes these itself
    assert 'export' not in kwargs
    assert 'use_external_tracking' not in kwargs
    # Parameters pyAFQ does not know are dropped with a warning naming them
    assert len(workflow_warnings) == 1
    assert workflow_warnings[0][-1] == 'not_a_pyafq_parameter, params_file'


@pytest.mark.parametrize(
    'spec_name',
    ['pyafq_tractometry', 'mrtrix_multishell_msmt_pyafq_tractometry'],
)
def test_builtin_specs_match_installed_pyafq(spec_name, workflow_warnings):
    """Every pyAFQ parameter in the built-in recon specs is recognized by the installed pyAFQ."""
    spec = yaml.safe_load(load_data.readable(f'pipelines/{spec_name}.yaml').read_text())
    pyafq_nodes = [node for node in spec['nodes'] if node['software'] == 'pyAFQ']
    assert len(pyafq_nodes) == 1

    kwargs = _parse_qsirecon_params_dict(pyafq_nodes[0]['parameters'])
    assert workflow_warnings == []
    # The spec keeps the pyAFQ 2 tractography and cleaning defaults explicit
    assert kwargs['tracking_params']['minlen'] == 50
    assert kwargs['tracking_params']['maxlen'] == 250
    assert kwargs['tracking_params']['trx'] is False
    assert kwargs['segmentation_params']['cleaning_params']['distance_threshold'] == 3


def test_init_pyafq_wf_requires_anatomical_data():
    """Workflow construction fails early when the preprocessed data have no anatomical image."""
    from qsirecon.workflows.recon.pyafq import init_pyafq_wf

    with pytest.raises(ValueError, match='requires an anatomical image'):
        init_pyafq_wf({'has_qsiprep_anat': False}, name='afq', params={})


def test_pyafq_interface_requires_pve_for_tractography(tmp_path):
    """The interface refuses to let pyAFQ track without tissue probability maps."""
    import nibabel as nib
    import numpy as np

    from qsirecon.interfaces.pyafq import PyAFQRecon

    img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    files = {}
    for name in ('dwi_file', 't1_file', 'mask_file'):
        files[name] = str(tmp_path / f'{name}.nii.gz')
        img.to_filename(files[name])

    for name in ('bval_file', 'bvec_file', 'itk_file'):
        files[name] = str(tmp_path / name)
        (tmp_path / name).write_text('0\n')

    interface = PyAFQRecon(kwargs={}, **files)
    interface.inputs.csf_probseg = None
    with pytest.raises(ValueError, match='probability maps'):
        interface.run(cwd=str(tmp_path))
