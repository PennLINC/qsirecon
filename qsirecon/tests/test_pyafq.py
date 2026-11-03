"""Tests for the pyAFQ recon spec parser."""

import pytest
import yaml

from qsirecon.data import load as load_data
from qsirecon.workflows.recon.pyafq import _parse_qsirecon_params_dict


@pytest.mark.parametrize(
    'spec_name',
    ['pyafq_tractometry', 'mrtrix_multishell_msmt_pyafq_tractometry'],
)
def test_pyafq_specs_use_known_params(spec_name):
    """Fail when a pyAFQ release removes a parameter that a built-in spec sets."""
    spec = yaml.safe_load(load_data.readable(f'pipelines/{spec_name}.yaml').read_text())
    node = next(node for node in spec['nodes'] if node['software'] == 'pyAFQ')
    _parse_qsirecon_params_dict(node['parameters'])


def test_pyafq_params_are_nested():
    kwargs = _parse_qsirecon_params_dict(
        {
            'n_seeds': 1000,
            'filter_by_endpoints': False,
            'clean_rounds': 2,
            'min_bval': '',
        }
    )
    assert kwargs == {
        'tracking_params': {'n_seeds': 1000},
        'segmentation_params': {
            'filter_by_endpoints': False,
            'cleaning_params': {'clean_rounds': 2},
        },
    }


def test_pyafq_rejects_unknown_params():
    with pytest.raises(ValueError, match='tracker'):
        _parse_qsirecon_params_dict({'tracker': 'local'})
