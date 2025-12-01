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
    assert fs_path.name == "sub-01"

    # If you pass in a valid path with subject ID that exists and session ID, you get the path
    (fs_subjects_dir / "sub-01_ses-01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.name == "sub-01_ses-01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="02",
    )
    assert fs_path.name == "sub-01"

    # If you pass in a valid path with subject ID that exists and session ID, you get the path
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.name == "sub-01_ses-01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["02", Query.NONE],
    )
    assert fs_path.name == "sub-01"

    (fs_subjects_dir / "sub-01_ses-01.long.sub-01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.name == "sub-01_ses-01.long.sub-01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.name == "sub-01_ses-01.long.sub-01"

    # Now without sub- and ses- prefixes.
    fs_subjects_dir = tmp_path_factory.mktemp("freesurfer_02")
    (fs_subjects_dir / "01").mkdir(parents=True)

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=None,
    )
    assert fs_path.name == "01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.name == "01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.name == "01"

    (fs_subjects_dir / "01_01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.name == "01_01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.name == "01_01"

    (fs_subjects_dir / "01_01.long.01").mkdir(parents=True)
    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id="01",
    )
    assert fs_path.name == "01_01.long.01"

    fs_path = find_fs_path(
        freesurfer_dir=fs_subjects_dir,
        subject_id="01",
        session_id=["01", Query.NONE],
    )
    assert fs_path.name == "01_01.long.01"


def test_deep_update_dict_basic():
    """Test basic dictionary update with deep_update_dict."""
    from qsirecon.utils.misc import deep_update_dict

    base = {"a": 1, "b": 2}
    update = {"b": 3, "c": 4}
    result = deep_update_dict(base, update)
    assert result == {"a": 1, "b": 3, "c": 4}
    # Verify it modifies in place
    assert base == {"a": 1, "b": 3, "c": 4}


def test_deep_update_dict_nested():
    """Test nested dictionary merging."""
    from qsirecon.utils.misc import deep_update_dict

    base = {
        "Model": {"param1": "a", "param2": "b"},
        "Other": {"x": 1},
    }
    update = {
        "Model": {"param2": "c", "param3": "d"},
        "Other": {"y": 2},
    }
    result = deep_update_dict(base, update)
    assert result == {
        "Model": {"param1": "a", "param2": "c", "param3": "d"},
        "Other": {"x": 1, "y": 2},
    }


def test_deep_update_dict_deeply_nested():
    """Test deeply nested dictionary merging."""
    from qsirecon.utils.misc import deep_update_dict

    base = {
        "level1": {
            "level2": {
                "level3": {"a": 1, "b": 2},
                "other": "value",
            }
        }
    }
    update = {
        "level1": {
            "level2": {
                "level3": {"b": 3, "c": 4},
            }
        }
    }
    result = deep_update_dict(base, update)
    assert result == {
        "level1": {
            "level2": {
                "level3": {"a": 1, "b": 3, "c": 4},
                "other": "value",
            }
        }
    }


def test_deep_update_dict_replace_non_dict():
    """Test that non-dict values are replaced."""
    from qsirecon.utils.misc import deep_update_dict

    # Replace dict with non-dict
    base = {"key": {"nested": "value"}}
    update = {"key": "simple_value"}
    result = deep_update_dict(base, update)
    assert result == {"key": "simple_value"}

    # Replace non-dict with dict
    base = {"key": "simple_value"}
    update = {"key": {"nested": "value"}}
    result = deep_update_dict(base, update)
    assert result == {"key": {"nested": "value"}}


def test_deep_update_dict_empty():
    """Test with empty dictionaries."""
    from qsirecon.utils.misc import deep_update_dict

    # Empty update
    base = {"a": 1, "b": {"c": 2}}
    update = {}
    result = deep_update_dict(base, update)
    assert result == {"a": 1, "b": {"c": 2}}

    # Empty base
    base = {}
    update = {"a": 1, "b": {"c": 2}}
    result = deep_update_dict(base, update)
    assert result == {"a": 1, "b": {"c": 2}}


def test_deep_update_dict_mixed_types():
    """Test with mixed types (lists, numbers, strings)."""
    from qsirecon.utils.misc import deep_update_dict

    base = {
        "dict_key": {"nested": "value"},
        "list_key": [1, 2, 3],
        "string_key": "hello",
        "number_key": 42,
    }
    update = {
        "dict_key": {"nested": "updated", "new": "value"},
        "list_key": [4, 5],  # Lists are replaced, not merged
        "string_key": "world",
        "number_key": 100,
    }
    result = deep_update_dict(base, update)
    assert result == {
        "dict_key": {"nested": "updated", "new": "value"},
        "list_key": [4, 5],
        "string_key": "world",
        "number_key": 100,
    }


def test_conditional_doc():
    """Test ConditionalDoc."""
    from nipype.interfaces.base import Undefined

    from qsirecon.utils.boilerplate import ConditionalDoc

    # Test with a value but no formatting
    doc = ConditionalDoc(if_true="A passing test.", if_false="A failing test.")
    assert doc.get_doc(value=True) == "A passing test."
    assert doc.get_doc(value=False) == "A failing test."
    # Undefined inherits from if_false when not specified.
    assert doc.get_doc(value=Undefined) == "A failing test."
    # Integers are fine.
    assert doc.get_doc(value=1) == "A passing test."
    # Only actual booleans are treated as such. 0 is not False.
    assert doc.get_doc(value=0) == "A passing test."
    assert doc.get_doc(value="test") == "A passing test."
    assert doc.get_doc(value="") == "A passing test."

    # Test with a value and formatting
    doc = ConditionalDoc(
        if_true="A passing test with value {value}.",
        if_false="A failing test with value {value}.",
        if_undefined="An undefined test with value {value}.",
    )
    assert doc.get_doc(value=True) == "A passing test with value True."
    assert doc.get_doc(value=False) == "A failing test with value False."
    assert doc.get_doc(value=Undefined) == "An undefined test with value <undefined>."
    # Integers are fine.
    assert doc.get_doc(value=1) == "A passing test with value 1."
    assert doc.get_doc(value=0) == "A passing test with value 0."
    # Floats are fine.
    assert doc.get_doc(value=0.5) == "A passing test with value 0.5."
    assert doc.get_doc(value=-0.5) == "A passing test with value -0.5."

    assert doc.get_doc(value="test") == "A passing test with value test."
    assert doc.get_doc(value="") == "A passing test with value ."


def test_build_documentation():
    """Test build_documentation."""
    import nipype.pipeline.engine as pe

    from qsirecon.interfaces.tortoise import EstimateTensor
    from qsirecon.utils.boilerplate import build_documentation

    # Test with an interface object
    interface = EstimateTensor(reg_mode="DIAG", free_water_diffusivity=3000, write_cs=False)
    doc = build_documentation(interface)
    assert doc == (
        "All b-values were used for tensor fitting. "
        "Free water diffusivity was set to 3000 (mu m)^2/s. "
        "Tensor fitting was performed with DIAG regularization."
    )

    # Test with a node
    node = pe.Node(
        EstimateTensor(reg_mode="DIAG", free_water_diffusivity=3000, write_cs=False),
        name="estimate_tensor",
    )
    doc = build_documentation(node)
    assert doc == (
        "All b-values were used for tensor fitting. "
        "Free water diffusivity was set to 3000 (mu m)^2/s. "
        "Tensor fitting was performed with DIAG regularization."
    )


def test_response_function_conversion(tmp_path_factory):
    """Test response function conversion."""
    import json

    import numpy as np

    from qsirecon.utils.misc import (
        bids_response_function_to_mrtrix,
        mrtrix_response_function_to_bids,
    )

    response_function = np.random.random((6, 5))
    txt_file = tmp_path_factory.mktemp("test_response_function_conversion") / "test.txt"
    with open(txt_file, "w") as f:
        np.savetxt(f, response_function)

    arr = mrtrix_response_function_to_bids(txt_file)
    assert np.allclose(arr, response_function)

    json_file = tmp_path_factory.mktemp("test_response_function_conversion") / "test.json"
    with open(json_file, "w") as f:
        json.dump({"ResponseFunction": {"Coefficients": arr}}, f)

    arr2 = bids_response_function_to_mrtrix(json_file)
    assert np.allclose(arr2, response_function)
