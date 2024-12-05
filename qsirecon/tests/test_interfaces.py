"""Command-line interface tests."""

import os

import pytest

from qsirecon.tests.utils import download_test_data


@pytest.mark.integration
@pytest.mark.interfaces
def test_shell_selection(data_dir, working_dir):
    """Run reconstruction workflow tests.


    """
    TEST_NAME = "mrtrix_singleshell_ss3t_act"

    dataset_dir = download_test_data("multishell_output", data_dir)
    dataset_dir = os.path.join(dataset_dir, "qsiprep")



