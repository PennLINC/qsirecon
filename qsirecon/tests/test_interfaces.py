"""Command-line interface tests."""

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from qsirecon.interfaces.gradients import GradientSelect
from qsirecon.tests.utils import download_test_data


@pytest.mark.integration
@pytest.mark.interfaces
def test_shell_selection(data_dir, working_dir):
    """Run reconstruction workflow tests."""
    dwi_prefix = "sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi"
    data_dir = "/home/matt/projects/qsirecon/.circleci/data"
    dataset_dir = Path(download_test_data("multishell_output", data_dir))
    dataset_dir = Path(dataset_dir) / "multishell_output" / "qsiprep"
    data_stem = str(dataset_dir / "sub-ABCD" / "dwi" / dwi_prefix)

    # Test the actual data (including a nifti)
    grad_select = GradientSelect(
        dwi_file=data_stem + ".nii.gz",
        bval_file=data_stem + ".bval",
        bvec_file=data_stem + ".bvec",
        b_file=data_stem + ".b",
        btable_file=data_stem + ".b",
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
