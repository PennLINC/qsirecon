#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utility functions."""


def check_deps(workflow):
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and which(node.interface._cmd.split()[0]) is None)
    )


def load_yaml(fname):
    import yaml

    with open(fname) as f:
        return yaml.safe_load(f)


def remove_non_alphanumeric(input_string: str) -> str:
    # Replace all non-alphanumeric characters with an empty string
    import re

    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return cleaned_string


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    This provides a simple way to accept flexible arguments.
    """
    import numpy as np

    return obj if isinstance(obj, (list, tuple, np.ndarray)) else [obj]


def deep_update_dict(base_dict, update_dict):
    """
    Recursively update a dictionary with another dictionary.

    This function updates base_dict with values from update_dict,
    recursively merging nested dictionaries instead of replacing them.

    Parameters
    ----------
    base_dict : dict
        The dictionary to be updated
    update_dict : dict
        The dictionary with updates

    Returns
    -------
    dict
        The updated dictionary
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            # Recursively update nested dictionaries
            deep_update_dict(base_dict[key], value)
        else:
            # Otherwise, set or overwrite the value
            base_dict[key] = value
    return base_dict


def bids_response_function_to_mrtrix(json_file):
    """Convert a JSON response function to an MRtrix format."""
    import json

    import numpy as np

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    return np.array(json_data['ResponseFunction']['Coefficients'])


def mrtrix_response_function_to_bids(response_function_file):
    """Load a response function from MRtrix3 and convert to JSON-compatible format."""
    import numpy as np

    response_data = np.loadtxt(response_function_file)
    if response_data.ndim == 1:
        return [[value] for value in response_data]
    return [row.tolist() for row in response_data]
