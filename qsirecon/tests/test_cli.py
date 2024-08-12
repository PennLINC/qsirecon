"""Command-line interface tests."""

import os
import sys
from unittest.mock import patch

import pytest
from nipype import config as nipype_config

from qsirecon.cli import run
from qsirecon.cli.parser import parse_args
from qsirecon.cli.workflow import build_boilerplate, build_workflow
from qsirecon.tests.utils import (
    check_generated_files,
    download_test_data,
    get_test_data_path,
)
from qsirecon.utils.bids import write_derivative_description
from qsirecon.viz.reports import generate_reports

nipype_config.enable_debug_mode()


@pytest.mark.integration
@pytest.mark.mrtrix_singleshell_ss3t_act
def test_mrtrix_singleshell_ss3t_act(data_dir, output_dir, working_dir):
    """Run reconstruction workflow tests.

    Was in 3TissueReconTests.sh. I split it between this and the multi-shell test.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - qsirecon single shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "mrtrix_singleshell_ss3t_act"

    dataset_dir = download_test_data("singleshell_output", data_dir)
    dataset_dir = os.path.join(dataset_dir, "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=mrtrix_singleshell_ss3t_ACT-fast",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.mrtrix_singleshell_ss3t_noact
def test_mrtrix_singleshell_ss3t_noact(data_dir, output_dir, working_dir):
    """Run reconstruction workflow tests.

    Was in 3TissueReconTests.sh. I split it between this and the single-shell test.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "mrtrix_singleshell_ss3t_noact"

    dataset_dir = download_test_data("singleshell_output", data_dir)
    dataset_dir = os.path.join(dataset_dir, "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=mrtrix_singleshell_ss3t_noACT",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.amico_noddi
def test_amico_noddi(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in AMICOReconTests.sh.
    All supported reconstruction workflows get tested.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - DSDTI BIDS data (data/singleshell_output)
    """
    TEST_NAME = "amico_noddi"

    dataset_dir = download_test_data("singleshell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=amico_noddi",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.autotrack
def test_autotrack(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in AutoTrackTest.sh.

    All supported reconstruction workflows get tested.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - DSDTI BIDS data (data/multishell_output)
    """
    TEST_NAME = "autotrack"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=dsi_studio_autotrack",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dipy_mapmri
def test_dipy_mapmri(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in DipyReconTests.sh. I split it between this and the dipy_dki test.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs:
    -------

    - qsirecon single shell results (data/DSDTI_fmap)
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "dipy_mapmri"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=dipy_mapmri",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dipy_dki
def test_dipy_dki(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in DipyReconTests.sh. I split it between this and the dipy_mapmri test.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs:
    -------

    - qsirecon single shell results (data/DSDTI_fmap)
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "dipy_dki"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=dipy_dki",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.scalar_mapper
def test_scalar_mapper(data_dir, output_dir, working_dir):
    """Test the TORTOISE recon workflow.

    All supported reconstruction workflows get tested

    Inputs
    ------
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "scalar_mapper"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=test_scalar_maps",
        "--output-resolution=3.5",
        "--nthreads=1",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.pyafq_recon_external_trk
def test_pyafq_recon_external_trk(data_dir, output_dir, working_dir):
    """Reconstruction workflow tests

    All supported reconstruction workflows get tested

    This tests the following features:
    - pyAFQ pipeline with tractography done in mrtrix

    Inputs
    ------
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "pyafq_recon_external_trk"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=mrtrix_multishell_msmt_pyafq_tractometry",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.pyafq_recon_full
def test_pyafq_recon_full(data_dir, output_dir, working_dir):
    """Reconstruction workflow tests

    All supported reconstruction workflows get tested

    This tests the following features:
    - Full pyAFQ pipeline

    Inputs
    ------
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "pyafq_recon_full"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=pyafq_tractometry",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.mrtrix3_recon
def test_mrtrix3_recon(data_dir, output_dir, working_dir):
    """Reconstruction workflow tests

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - qsirecon single shell results (data/DSDTI_fmap)
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "mrtrix3_recon"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=mrtrix_multishell_msmt_ACT-fast",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.tortoise_recon
def test_tortoise_recon(data_dir, output_dir, working_dir):
    """Test the TORTOISE recon workflow

    All supported reconstruction workflows get tested

    Inputs
    ------
    - qsirecon multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "tortoise_recon"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--recon-spec=TORTOISE",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


def _run_and_generate(test_name, parameters, test_main=True):
    from qsirecon import config

    # TODO: Add --clean-workdir param to CLI
    parameters.append("--stop-on-first-crash")
    parameters.append("--notrack")
    parameters.append("-vv")

    if test_main:
        # This runs, but for some reason doesn't count toward coverage.
        argv = ["qsirecon"] + parameters
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as e:
                run.main()

            assert e.value.code == 0
    else:
        # XXX: This isn't working because config.execution.fs_license_file is None.
        parse_args(parameters)
        config_file = config.execution.work_dir / f"config-{config.execution.run_uuid}.toml"
        config.loggers.cli.warning(f"Saving config file to {config_file}")
        config.to_filename(config_file)

        retval = build_workflow(config_file, retval={})
        qsirecon_wf = retval["workflow"]
        qsirecon_wf.run()
        write_derivative_description(config.execution.fmri_dir, config.execution.output_dir)

        build_boilerplate(str(config_file), qsirecon_wf)
        session_list = (
            config.execution.bids_filters.get("bold", {}).get("session")
            if config.execution.bids_filters
            else None
        )
        generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.output_dir,
            run_uuid=config.execution.run_uuid,
            session_list=session_list,
        )

    output_list_file = os.path.join(get_test_data_path(), f"{test_name}_outputs.txt")
    optional_outputs_list = os.path.join(get_test_data_path(), f"{test_name}_optional_outputs.txt")
    if not os.path.isfile(optional_outputs_list):
        optional_outputs_list = None

    check_generated_files(config.execution.output_dir, output_list_file, optional_outputs_list)
