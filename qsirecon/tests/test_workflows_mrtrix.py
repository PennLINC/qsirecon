"""Tests for qsirecon.workflows.recon.mrtrix."""

import pytest

from qsirecon import config
from qsirecon.workflows.recon.mrtrix import init_mrtrix_csd_recon_wf


@pytest.fixture
def mrtrix_config(tmp_path, monkeypatch):
    """Set the config values the MRtrix CSD workflow builder reads."""
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)
    monkeypatch.setattr(config.execution, 'output_dir', tmp_path)


def _build_precomputed_wf(fod_algorithm, response):
    params = {
        'fod': {'algorithm': fod_algorithm},
        'response': {'algorithm': 'precomputed', **response},
    }
    return init_mrtrix_csd_recon_wf(
        {'has_qsiprep_t1w_transforms': False},
        name='csd',
        qsirecon_suffix='MRtrix3',
        params=params,
    )


def test_precomputed_single_tissue_needs_only_wm(mrtrix_config):
    """Single-tissue CSD with a precomputed WM response does not ask for GM or CSF files."""
    wf = _build_precomputed_wf('csd', {'wm_txt': '/responses/wm.txt'})

    loader = wf.get_node('load_response_functions')
    assert loader.inputs.using_multitissue is False
    assert loader.inputs.wm_file == '/responses/wm.txt'

    # There are no GM or CSF responses to write out.
    assert wf.get_node('ds_wm_txt') is not None
    assert wf.get_node('ds_gm_txt') is None
    assert wf.get_node('ds_csf_txt') is None


def test_precomputed_multi_tissue_needs_all_tissues(mrtrix_config):
    """Multi-tissue CSD with a precomputed response fails at build time without GM and CSF."""
    with pytest.raises(ValueError, match='require "gm_txt"'):
        _build_precomputed_wf('msmt_csd', {'wm_txt': '/responses/wm.txt'})

    wf = _build_precomputed_wf(
        'msmt_csd',
        {
            'wm_txt': '/responses/wm.txt',
            'gm_txt': '/responses/gm.txt',
            'csf_txt': '/responses/csf.txt',
        },
    )
    loader = wf.get_node('load_response_functions')
    assert loader.inputs.using_multitissue is True
    assert wf.get_node('ds_gm_txt') is not None
    assert wf.get_node('ds_csf_txt') is not None


def test_precomputed_relative_path_rejected(mrtrix_config):
    """Relative precomputed response paths fail at build time."""
    with pytest.raises(ValueError, match='must be an absolute path'):
        _build_precomputed_wf('csd', {'wm_txt': 'wm.txt'})
