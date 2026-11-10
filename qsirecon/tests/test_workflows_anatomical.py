"""Tests for qsirecon.workflows.recon.anatomical."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from nipype.interfaces.base import isdefined

from qsirecon import config
from qsirecon.interfaces.interchange import (
    anatomical_workflow_outputs,
    qsiprep_output_names,
)
from qsirecon.tests.utils import download_test_data, get_nodes
from qsirecon.utils.atlases import collect_atlases
from qsirecon.workflows.recon.anatomical import (
    HSV_REQUIREMENTS,
    init_dwi_recon_anatomical_workflow,
    init_highres_recon_anatomical_wf,
    init_warp_atlases_wf,
)

ATLASES = ['AAL116', 'AICHA384Ext']
QSIRECON_ATLASES = '/home/qsirecon/.cache/qsirecon/QSIReconAtlases'


def _unconnected_mandatory_inputs(wf, provided_inputs):
    """List mandatory node inputs that are neither set nor connected.

    Parameters
    ----------
    wf : Workflow
        The workflow to check.
    provided_inputs : collection of str
        The fields of the workflow's own ``inputnode`` that the parent workflow will
        supply. A connection from any other inputnode field is treated as undefined,
        because that is what it will be at runtime.
    """
    graph = wf._create_flat_graph()
    inputnode_name = f'{wf.name}.inputnode'
    missing = []
    for node in graph.nodes():
        connected = set()
        for src_node, _, data in graph.in_edges(node, data=True):
            for src, dst in data['connect']:
                src_field = src[0] if isinstance(src, tuple) else src
                if src_node.fullname == inputnode_name and src_field not in provided_inputs:
                    continue
                connected.add(dst)

        for name in node.inputs.traits(mandatory=True):
            if name in connected or isdefined(getattr(node.inputs, name)):
                continue
            missing.append(f'{node.fullname}.{name}')

    return sorted(missing)


def _node_basenames(wf):
    """The last component of every node name in a workflow."""
    return {name.split('.')[-1] for name in wf.list_node_names()}


@pytest.fixture
def build_config(tmp_path, monkeypatch):
    """Set the config values that the anatomical workflow builders read."""
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    monkeypatch.setattr(config.execution, 'output_dir', out_dir)
    monkeypatch.setattr(config.execution, 'fs_subjects_dir', None)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.workflow, 'output_resolution', None)
    monkeypatch.setattr(config.workflow, 'infant', False)
    return tmp_path


def _fake_freesurfer_dir(root, subject_id):
    """Create an empty FreeSurfer subject directory with the files HSVS needs."""
    subjects_dir = root / 'freesurfer'
    subject_dir = subjects_dir / f'sub-{subject_id}'
    for requirement in HSV_REQUIREMENTS:
        path = subject_dir / requirement
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return subjects_dir


NO_ANAT_STATUS = {
    'has_qsiprep_anat': False,
    'has_qsiprep_anat_transforms': False,
    'template_output_space': None,
}


def test_dwi_anat_wf_no_anatomical(build_config):
    """With no anatomical image and no FreeSurfer, the DWI is masked from its own b=0s.

    This is the ``--anat-modality none`` case. The template grid cannot be built, so it is
    skipped, and every remaining node must have its mandatory inputs connected.
    """
    wf, status = init_dwi_recon_anatomical_workflow(
        atlas_configs={},
        has_qsiprep_5tt_hsvs=False,
        needs_anat_transform=False,
        has_freesurfer_5tt_hsvs=False,
        has_qsiprep_anat=False,
        has_qsiprep_anat_transforms=False,
        has_freesurfer=False,
        extras_to_make=[],
        name='dwi_anat_wf',
        template_output_space=None,
    )
    nodes = _node_basenames(wf)
    assert 'mask_b0s' in nodes
    assert 'get_template' not in nodes
    assert 'resample_mask' not in nodes
    # Only the DWI data are available from the parent workflow.
    assert _unconnected_mandatory_inputs(wf, qsiprep_output_names) == []
    assert status['has_qsiprep_anat'] is False
    assert status['template_output_space'] is None


def test_dwi_anat_wf_with_template(build_config):
    """With a preprocessed anatomical image and transforms, the template grid is built."""
    wf, status = init_dwi_recon_anatomical_workflow(
        atlas_configs={},
        has_qsiprep_5tt_hsvs=False,
        needs_anat_transform=False,
        has_freesurfer_5tt_hsvs=False,
        has_qsiprep_anat=True,
        has_qsiprep_anat_transforms=True,
        has_freesurfer=False,
        extras_to_make=[],
        name='dwi_anat_wf',
        template_output_space='MNI152NLin2009cAsym',
    )
    nodes = _node_basenames(wf)
    assert 'get_template' in nodes
    assert 'resample_mask' in nodes
    assert 'odf_rois' in nodes
    assert 'mask_b0s' not in nodes
    provided = qsiprep_output_names + anatomical_workflow_outputs
    assert _unconnected_mandatory_inputs(wf, provided) == []
    assert status['template_output_space'] == 'MNI152NLin2009cAsym'


def test_anat_wfs_freesurfer_only_hsvs(build_config, monkeypatch):
    """FreeSurfer without a preprocessed anatomical image, with a HSVS 5tt requested.

    The subject-level workflow must forward the native-space FreeSurfer files without
    writing anatomical-named derivatives, and the DWI-level workflow must register them
    to the DWI and align the 5tt image with that transform.
    """
    subjects_dir = _fake_freesurfer_dir(build_config, '01')
    monkeypatch.setattr(config.execution, 'fs_subjects_dir', subjects_dir)

    highres_wf, status = init_highres_recon_anatomical_wf(
        subject_id='01',
        session_id=None,
        extras_to_make=['mrtrix_5tt_hsvs'],
        status=dict(NO_ANAT_STATUS),
        name='recon_anatomical_wf',
    )
    assert status['has_freesurfer'] is True
    assert status['has_freesurfer_5tt_hsvs'] is True
    assert status['has_qsiprep_5tt_hsvs'] is False
    nodes = _node_basenames(highres_wf)
    assert 'create_5tt_hsvs' in nodes
    assert 'fs_source' in nodes
    # These are named after an anatomical image, which does not exist here.
    assert 'ds_fs_5tt_hsvs' not in nodes
    assert 'ds_qsiprep_5tt_hsvs' not in nodes
    subworkflows = {n.split('.')[0] for n in highres_wf.list_node_names()}
    assert 'register_fs_to_qsiprep_wf' not in subworkflows
    # Nothing was collected from the preprocessed dataset, so no inputnode field is defined.
    assert _unconnected_mandatory_inputs(highres_wf, []) == []

    dwi_wf, dwi_status = init_dwi_recon_anatomical_workflow(
        atlas_configs={},
        needs_anat_transform=False,
        extras_to_make=['mrtrix_5tt_hsvs'],
        name='dwi_anat_wf',
        **status,
    )
    nodes = _node_basenames(dwi_wf)
    assert 'get_template' not in nodes
    assert 'register_to_qsiprep' in nodes
    assert 'apply_header_to_5tt_hsvs' in nodes
    assert 'ds_qsiprep_5tt_hsvs' in nodes
    # The subject-level workflow provides only the native-space FreeSurfer files and the
    # 5tt image; acpc_preproc, acpc_brain_mask and the transforms are undefined.
    provided = qsiprep_output_names + ['brain', 'aseg', 'fs_5tt_hsvs']
    assert _unconnected_mandatory_inputs(dwi_wf, provided) == []
    # FreeSurfer's brain and aseg stand in for the preprocessed anatomical image and mask.
    assert dwi_status['has_qsiprep_anat'] is True
    assert dwi_status['has_qsiprep_anat_transforms'] is False


def _mrtrix_grid(in_file):
    """Describe the voxel grid MRtrix sees for an image, as ``(size, transform)``.

    ``mrinfo`` reports both in MRtrix's canonical (stride-normalized) form, so two files
    covering the same physical grid give the same answer whether they are NIfTIs or MIFs.

    XXX: It would be nice to replace mrinfo with nibabel when that's possible.
    """

    def _mrinfo(flag):
        proc = subprocess.run(
            ['mrinfo', flag, str(in_file)],
            capture_output=True,
            check=True,
            text=True,
        )
        return np.array(proc.stdout.split(), dtype=float)

    # Drop any non-spatial axis so 4D DWIs can be compared against 3D atlases.
    return _mrinfo('-size')[:3].astype(int), _mrinfo('-transform').reshape(4, 4)


def _same_grid(grid1, grid2):
    """Check whether two :func:`_mrtrix_grid` results describe the same voxel grid."""
    return np.array_equal(grid1[0], grid2[0]) and np.allclose(grid1[1], grid2[1], atol=1e-3)


def test_warp_atlases_wf(data_dir, tmp_path_factory):
    """Test init_warp_atlases_wf.

    This should ensure that template-space atlases are warped to ACPC space
    and that all of the outputs (especially the MIF outputs) reflect this.

    We need (1) an ACPC-space DWI file, (2) a template-to-ACPC transform,
    and (3) at least one template-space atlas with associated files.
    """
    tmpdir = tmp_path_factory.mktemp('test_warp_atlases_wf')
    out_dir = tmpdir / 'out'

    dataset_dir = Path(download_test_data('multishell_output', data_dir))
    dataset_dir = Path(dataset_dir) / 'multishell_output' / 'qsiprep'
    xfm = (
        dataset_dir
        / 'sub-ABCD'
        / 'anat'
        / 'sub-ABCD_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
    )
    src = (
        dataset_dir
        / 'sub-ABCD'
        / 'dwi'
        / 'sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi.nii.gz'
    )

    # Collect exemplar atlases
    atlas_configs = collect_atlases(
        datasets={'qsireconatlases': QSIRECON_ATLASES},
        atlases=ATLASES,
        bids_filters={},
    )

    # The template and ACPC grids must differ, or the assertions below can't tell them apart.
    acpc_grid = _mrtrix_grid(src)
    template_grids = {name: _mrtrix_grid(atlas_configs[name]['image']) for name in ATLASES}
    for atlas_name, template_grid in template_grids.items():
        assert not _same_grid(template_grid, acpc_grid), (
            f'{atlas_name} is already on the DWI grid, so this test proves nothing'
        )

    wf = init_warp_atlases_wf(atlas_configs=atlas_configs)
    wf.base_dir = str(tmpdir)
    wf.inputs.inputnode.source_file = str(src)
    wf.inputs.inputnode.template_to_acpc_xfm = str(xfm)
    # The workflow leaves the datasinks' base_directory to the parent workflow's
    # clean_datasinks() call, so point them at a temporary derivatives directory here.
    for node_name in wf.list_node_names():
        if node_name.split('.')[-1].startswith('ds_'):
            wf.get_node(node_name).inputs.base_directory = str(out_dir)

    wf_res = wf.run()
    wf_nodes = get_nodes(wf_res)

    # The recombined atlas configs are what the downstream reconstruction workflows use.
    out_configs = wf_nodes['warp_atlases_wf.recombine_atlas_configs'].get_output('atlas_configs')
    assert sorted(out_configs.keys()) == sorted(ATLASES)

    for atlas_name in ATLASES:
        nifti_file = out_configs[atlas_name]['dwi_resolution_file']
        mif_file = out_configs[atlas_name]['dwi_resolution_mif']
        assert Path(nifti_file).is_file()
        assert Path(mif_file).is_file()

        # ApplyTransforms resamples the NIfTI atlas onto the DWI grid.
        nifti_grid = _mrtrix_grid(nifti_file)
        assert _same_grid(nifti_grid, acpc_grid), (
            f'{atlas_name} NIfTI grid {nifti_grid} != DWI grid {acpc_grid}'
        )

        # The MIF is what tck2connectome consumes, so it has to be on that grid too (#399).
        mif_grid = _mrtrix_grid(mif_file)
        assert _same_grid(mif_grid, acpc_grid), (
            f'{atlas_name} MIF grid {mif_grid} != DWI grid {acpc_grid}'
        )
        assert not _same_grid(mif_grid, template_grids[atlas_name]), (
            f'{atlas_name} MIF is still on the template grid {template_grids[atlas_name]}'
        )
