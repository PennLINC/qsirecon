#!/usr/bin/env python
"""
Standalone script to facilitate caching of required TemplateFlow templates.

QSIRecon fetches templates dynamically based on ``template_output_space``,
but MNI152NLin2009cAsym (adult default) and MNIInfant (infant mode) are the
most common choices.  Pre-caching them avoids network access at runtime.
"""

import argparse
import os

import templateflow.api as tf


def fetch_MNI2009():
    """MNI152NLin2009cAsym — default adult template."""
    template = 'MNI152NLin2009cAsym'
    tf.get(template, resolution=1, desc=None, suffix='T1w')
    tf.get(template, resolution=1, desc='brain', suffix='mask')


def fetch_MNIInfant():
    """MNIInfant — used when --infant is set."""
    tf.get('MNIInfant', resolution=1, desc=None, suffix='T1w')
    tf.get('MNIInfant', resolution=1, desc='brain', suffix='mask')


def fetch_all():
    fetch_MNI2009()
    fetch_MNIInfant()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Helper script for pre-caching required templates to run QSIRecon',
    )
    parser.add_argument(
        '--tf-dir',
        type=os.path.abspath,
        help=(
            'Directory to save templates in. '
            'If not provided, templates will be saved to `${HOME}/.cache/templateflow`.'
        ),
    )
    opts = parser.parse_args()

    if opts.tf_dir is not None:
        os.environ['TEMPLATEFLOW_HOME'] = opts.tf_dir

    fetch_all()
