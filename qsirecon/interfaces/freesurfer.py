#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""
import os.path as op
from pathlib import Path


def find_fs_path(freesurfer_dir, subject_id, session_id=None):
    """Find a freesurfer dir for subject or subject+session."""

    if freesurfer_dir is None:
        return None

    # Try first with session, if specified
    if session_id is not None:
        nosub = op.join(freesurfer_dir, f"{subject_id}_{session_id}")
        if op.exists(nosub):
            return Path(nosub)
        withsub = op.join(freesurfer_dir, f"sub-{subject_id}_ses-{session_id}")
        if op.exists(withsub):
            return Path(withsub)

    nosub = op.join(freesurfer_dir, subject_id)
    if op.exists(nosub):
        return Path(nosub)
    withsub = op.join(freesurfer_dir, f"sub-{subject_id}")
    if op.exists(withsub):
        return Path(withsub)
    return None
