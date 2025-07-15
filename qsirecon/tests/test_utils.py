"""Test utility functions."""

import os
from pprint import pprint

from bids.layout import BIDSLayout, Query
from niworkflows.utils.testing import generate_bids_skeleton

import qsirecon.utils.bids as xbids
from qsirecon.data import load as load_data


def test_collect_anatomical_data(tmp_path_factory):
    """Test collect_anatomical_data."""
    skeleton = load_data('tests/skeletons/longitudinal_anat.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_anatomical_data') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))

    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives'],
    )
    anat_input_filename = os.path.join(
        bids_dir,
        'sub-01',
        'anat',
        'sub-01_space-ACPC_desc-preproc_T1w.nii.gz',
    )
    anat_input_file = layout.get_file(anat_input_filename)

    if anat_input_file is None:
        _session_filter = None
    elif "session" not in anat_input_file.entities:
        _session_filter = Query.NONE
    else:
        _session_filter = [anat_input_file.entities["session"], Query.NONE]

    anat_data, highres_anat_status = xbids.collect_anatomical_data(
        layout=layout,
        subject_id='01',
        session_id=_session_filter,
        needs_t1w_transform=False,
        infant_mode=False,
        bids_filters={},
    )
    pprint(anat_data)
    pprint(highres_anat_status)
    assert highres_anat_status['has_qsiprep_t1w'] is True
    assert highres_anat_status['has_qsiprep_t1w_transforms'] is False
    assert anat_data['acpc_preproc'] is not None
    assert anat_data['acpc_brain_mask'] is not None
    assert anat_data['acpc_to_template_xfm'] is not None
    assert anat_data['template_to_acpc_xfm'] is not None
    assert anat_data['acpc_aseg'] is not None
    assert anat_data['acpc_brain'] is not None
    assert anat_data['acpc_preproc_xfm'] is not None
