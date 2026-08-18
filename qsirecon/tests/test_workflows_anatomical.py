"""Tests for qsirecon.workflows.recon.anatomical."""

import subprocess
from pathlib import Path

import numpy as np

from qsirecon.tests.utils import download_test_data, get_nodes
from qsirecon.utils.atlases import collect_atlases
from qsirecon.workflows.recon.anatomical import init_warp_atlases_wf

ATLASES = ['AAL116', 'AICHA384Ext']
QSIRECON_ATLASES = '/home/qsirecon/.cache/qsirecon/QSIReconAtlases'


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
