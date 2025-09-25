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

from bids.utils import listify


def find_fs_path(freesurfer_dir, subject_id, session_id=None):
    """Find a freesurfer dir for subject or subject+session.

    Parameters
    ----------
    freesurfer_dir : str
        The directory containing the freesurfer outputs.
    subject_id : str
        The subject ID.
    session_id : list or str or None, optional
        The session ID.
        May be None, a string, or a list containing strings and Query.NONE objects.

    Returns
    -------
    path : Path or None
        The path to the freesurfer directory.
    """

    if freesurfer_dir is None:
        return None

    # Look for longitudinal pipeline outputs first
    # Since session_id can be a two-item list with a session and a Query.NONE,
    # or just a string with the session, or None, we need to extract the actual session ID
    # robustly, when it's present.
    session_id = listify(session_id)
    session_id = [ses_id for ses_id in session_id if isinstance(ses_id, str)]
    if session_id:
        # There should only be one actual session ID
        session_id = session_id[0]

        nosub = op.join(freesurfer_dir, f"{subject_id}_{session_id}.long.{subject_id}")
        if op.exists(nosub):
            return Path(nosub)

        withsub = op.join(
            freesurfer_dir,
            f"sub-{subject_id}_ses-{session_id}.long.sub-{subject_id}",
        )
        if op.exists(withsub):
            return Path(withsub)

        # Next try with session but not longitudinal processing, if specified
        nosub = op.join(freesurfer_dir, f"{subject_id}_{session_id}")
        if op.exists(nosub):
            return Path(nosub)

        withsub = op.join(freesurfer_dir, f"sub-{subject_id}_ses-{session_id}")
        if op.exists(withsub):
            return Path(withsub)

    # Then look for cross-sectional pipeline outputs
    nosub = op.join(freesurfer_dir, subject_id)
    if op.exists(nosub):
        return Path(nosub)

    withsub = op.join(freesurfer_dir, f"sub-{subject_id}")
    if op.exists(withsub):
        return Path(withsub)

    return None
