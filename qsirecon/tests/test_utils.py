"""Test utility functions."""

import os

from bids.layout import BIDSLayout, Query
from niworkflows.utils.testing import generate_bids_skeleton

import qsirecon.utils.bids as xbids
from qsirecon.data import load as load_data


def test_collect_anatomical_data(tmp_path_factory):
    """Test collect_anatomical_data."""
    skeleton = load_data("tests/skeletons/longitudinal_anat.yml")
    bids_dir = tmp_path_factory.mktemp("test_collect_anatomical_data") / "bids"
    generate_bids_skeleton(str(bids_dir), str(skeleton))

    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=["bids", "derivatives"],
    )
    anat_input_filename = os.path.join(
        bids_dir,
        "sub-01",
        "anat",
        "sub-01_space-ACPC_desc-preproc_T1w.nii.gz",
    )
    anat_input_file = layout.get_file(anat_input_filename)

    if anat_input_file is None:
        _session_filter = None
    elif "session" not in anat_input_file.entities:
        _session_filter = Query.NONE
    else:
        # Session-wise anatomical processing
        _session_filter = [anat_input_file.entities["session"], Query.NONE]

    anat_data, highres_anat_status = xbids.collect_anatomical_data(
        layout=layout,
        subject_id="01",
        session_id=_session_filter,
        needs_t1w_transform=False,
        infant_mode=False,
        bids_filters={},
    )
    assert highres_anat_status["has_qsiprep_t1w"] is True
    assert highres_anat_status["has_qsiprep_t1w_transforms"] is True
    # These are collected
    assert anat_data["acpc_preproc"] is not None
    assert anat_data["acpc_brain_mask"] is not None
    assert anat_data["acpc_to_template_xfm"] is not None
    assert anat_data["template_to_acpc_xfm"] is not None
    assert anat_data["acpc_aseg"] is not None
    # A number of inputs are not collected
    assert anat_data["acpc_aparc"] is None
    assert anat_data["acpc_csf_probseg"] is None
    assert anat_data["acpc_gm_probseg"] is None
    assert anat_data["acpc_wm_probseg"] is None
    # This should be collected, but is not because the file is in a session folder.
    assert anat_data["orig_to_acpc_xfm"] is None


def test_get_iterable_dwis_and_anats(tmp_path_factory):
    """Test get_iterable_dwis_and_anats."""
    skeleton = load_data("tests/skeletons/longitudinal_anat.yml")
    bids_dir = tmp_path_factory.mktemp("test_get_iterable_dwis_and_anats") / "bids"
    generate_bids_skeleton(str(bids_dir), str(skeleton))

    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=["bids", "derivatives"],
    )
    dwis_and_anats = xbids.get_iterable_dwis_and_anats(layout=layout)
    assert len(dwis_and_anats) == 1
    assert dwis_and_anats[0][0] is not None
    assert dwis_and_anats[0][1] is not None


def test_find_fs_path(tmp_path_factory):
    """Test find_fs_path."""
    from qsirecon.interfaces.freesurfer import find_fs_path

    fs_subjects_dir = tmp_path_factory.mktemp("freesurfer_01")
    (fs_subjects_dir / "sub-01").mkdir(parents=True)

    # If you pass in None, you get None
    assert find_fs_path(freesurfer_dir=None, subject_id="01", session_id=None) is None

    # If you pass in a valid path with subject ID that doesn't exist, you get None
    assert find_fs_path(freesurfer_dir=fs_subjects_dir, subject_id="fail", session_id=None) is None

    # If you pass in a valid path with subject ID that exists, you get the path
    fs_path = find_fs_path(freesurfer_dir=fs_subjects_dir, subject_id="01", session_id=None)
    assert fs_path.basename == "sub-01"

    # If you pass in a valid path with subject ID that exists and session ID, you get the path
    (fs_subjects_dir / "sub-01_ses-01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.basename == "sub-01_ses-01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="02",
    )
    assert fs_path.basename == "sub-01"

    # If you pass in a valid path with subject ID that exists and session ID, you get the path
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.basename == "sub-01_ses-01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["02", Query.NONE],
    )
    assert fs_path.basename == "sub-01"

    (fs_subjects_dir / "sub-01_ses-01.long.sub-01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.basename == "sub-01_ses-01.long.sub-01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.basename == "sub-01_ses-01.long.sub-01"

    # Now without sub- and ses- prefixes.
    fs_subjects_dir = tmp_path_factory.mktemp("freesurfer_02")
    (fs_subjects_dir / "01").mkdir(parents=True)

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=None,
    )
    assert fs_path.basename == "01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.basename == "01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.basename == "01"

    (fs_subjects_dir / "01_01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.basename == "01_01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.basename == "01_01"

    (fs_subjects_dir / "01_01.long.01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.basename == "01_01.long.01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.basename == "01_01.long.01"
