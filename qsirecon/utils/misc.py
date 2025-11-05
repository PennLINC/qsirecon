#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utility functions."""


def check_deps(workflow):
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, "_cmd") and which(node.interface._cmd.split()[0]) is None)
    )


def load_yaml(fname):
    import yaml

    with open(fname) as f:
        return yaml.safe_load(f)


def remove_non_alphanumeric(input_string: str) -> str:
    # Replace all non-alphanumeric characters with an empty string
    import re

    cleaned_string = re.sub(r"[^a-zA-Z0-9]", "", input_string)
    return cleaned_string


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    This provides a simple way to accept flexible arguments.
    """
    import numpy as np

    return obj if isinstance(obj, (list, tuple, np.ndarray)) else [obj]
