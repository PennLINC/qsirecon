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


def find_fs_path(freesurfer_dir, subject_id):
    if freesurfer_dir is None:
        return None
    nosub = op.join(freesurfer_dir, subject_id)
    if op.exists(nosub):
        return Path(nosub)
    withsub = op.join(freesurfer_dir, "sub-" + subject_id)
    if op.exists(withsub):
        return Path(withsub)
    return None
