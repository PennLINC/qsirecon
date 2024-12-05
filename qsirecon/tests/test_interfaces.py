"""Command-line interface tests."""

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from qsirecon.data import load as load_data
from qsirecon.interfaces.gradients import GradientSelect, _find_shells
from qsirecon.tests.utils import download_test_data, get_test_data_path


@pytest.mark.interfaces
def test_shell_selection(data_dir, working_dir):
    """Run reconstruction workflow tests."""
    dwi_prefix = "sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi"
    data_dir = "/home/matt/projects/qsirecon/.circleci/data"
    dataset_dir = Path(download_test_data("multishell_output", data_dir))
    dataset_dir = Path(dataset_dir) / "multishell_output" / "qsiprep"
    data_stem = str(dataset_dir / "sub-ABCD" / "dwi" / dwi_prefix)

    # Test actual data (including a nifti)
    grad_select = GradientSelect(
        dwi_file=data_stem + ".nii.gz",
        bval_file=data_stem + ".bval",
        bvec_file=data_stem + ".bvec",
        b_file=data_stem + ".b",
        bval_distance_cutoff=100,
        expected_n_input_shells=5,
        requested_shells=[0, "lowest", "highest"],
    )
    grad_select.run()
    correct_n = 73
    sel_nii = nb.load(dwi_prefix + "_selected.nii.gz")
    assert sel_nii.shape[3] == correct_n
    sel_bval = np.loadtxt(dwi_prefix + "_selected.bval")
    assert sel_bval.shape == (correct_n,)
    sel_bvec = np.loadtxt(dwi_prefix + "_selected.bvec")
    assert sel_bvec.shape == (3, correct_n)
    sel_b = np.loadtxt(dwi_prefix + "_selected.b")
    assert sel_b.shape[0] == correct_n
    # There is no btable from this dataset because it was created
    # before those were written in the outputs.
    assert not Path(dwi_prefix + "_selected.txt").exists()

    # Check some other sequences we might run into
    # ABCD has 4 shells + b=0s
    abcd_bval = np.loadtxt(load_data("schemes/ABCD.bval"))
    abcd_df = _find_shells(abcd_bval, 100)
    assert abcd_df["shell_num"].max() == 5.0

    # HCP has 3 shells + b=0s
    hcp_bval = np.loadtxt(load_data("schemes/HCP.bval"))
    hcp_df = _find_shells(hcp_bval, 100)
    assert hcp_df["shell_num"].max() == 4.0

    # DSIQ5 should raise an exception
    dsi_bval = np.loadtxt(load_data("schemes/DSIQ5.bval"))
    with pytest.raises(Exception):
        _find_shells(dsi_bval, 100)

    # Some other assorted test schemes that should fail
    nonshelled_schemes = ["HASC55-1", "HASC55-2", "HASC92", "RAND57"]
    for scheme in nonshelled_schemes:
        bval_file = Path(get_test_data_path()) / f"{scheme}.bval"
        test_bvals = np.loadtxt(bval_file)

        with pytest.raises(Exception):
            _find_shells(test_bvals, 100)
