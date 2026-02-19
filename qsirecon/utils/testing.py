# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for testing the workflows module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import logging

import pytest
from networkx.exception import NetworkXUnfeasible
from nipype.interfaces import utility as niu
from nipype.interfaces.base import isdefined
from nipype.pipeline import engine as pe

logging.disable(logging.INFO)  # <- do we really want to do this?


def assert_is_almost_expected_workflow(
    expected_name, expected_interfaces, expected_inputs, expected_outputs, actual
):
    """Somewhat hacky way to confirm workflows are as expected, but with low confidence."""
    assert isinstance(actual, pe.Workflow)
    assert expected_name == actual.name

    actual_nodes = [actual.get_node(name) for name in actual.list_node_names()]
    actual_ifaces = [node.interface.__class__.__name__ for node in actual_nodes]

    assert_is_subset_of_list(expected_interfaces, actual_ifaces)
    assert_is_subset_of_list(actual_ifaces, expected_interfaces)

    actual_inputs, actual_outputs = get_inputs_outputs(actual_nodes)

    assert_is_subset_of_list(expected_outputs, actual_outputs)
    assert_is_subset_of_list(expected_inputs, actual_inputs)


def assert_is_subset_of_list(expecteds, actuals):
    for expected in expecteds:
        assert expected in actuals


def get_inputs_outputs(nodes):
    def get_io_names(pre, ios):
        return [pre + str(io[0]) for io in ios]

    actual_inputs = []
    actual_outputs = []
    node_tuples = [(node.name, node.inputs.items(), node.outputs.items()) for node in nodes]
    for name, inputs, outputs in node_tuples:
        pre = str(name) + '.'
        actual_inputs += get_io_names(pre, inputs)

        pre = pre if pre[0:-1] != 'inputnode' else ''
        actual_outputs += get_io_names(pre, outputs)

    return actual_inputs, actual_outputs


def assert_circular(workflow, circular_connections):
    """Check key paths in workflow by specifying connections that should induce circular paths.

    circular_connections is a list of tuples::

        [('from_node_name', 'to_node_name', ('from_node.output_field', 'to_node.input_field'))]
    """
    for from_node, to_node, fields in circular_connections:
        from_node = workflow.get_node(from_node)
        to_node = workflow.get_node(to_node)
        workflow.connect([(from_node, to_node, fields)])

        with pytest.raises(NetworkXUnfeasible):
            workflow.write_graph()

        workflow.disconnect([(from_node, to_node, fields)])


def assert_inputs_set(workflow, additional_inputs=None):
    """Check that all mandatory inputs of nodes in the workflow (at the first level) are set.

    Additionally, check that inputs in *additional_inputs* are set.  An input is
    "set" if it is either defined explicitly (e.g. in the Interface declaration)
    or connected to another node's output (e.g. using ``workflow.connect``).

    Parameters
    ----------
    workflow : pe.Workflow
    additional_inputs : dict, optional
        ``{'node_name': ['mandatory', 'input', 'fields']}``
    """
    if additional_inputs is None:
        additional_inputs = {}

    dummy_node = pe.Node(niu.IdentityInterface(fields=['dummy']), name='DummyNode')
    node_names = [name for name in workflow.list_node_names() if name.count('.') == 0]
    for node_name in set(node_names + list(additional_inputs.keys())):
        node = workflow.get_node(node_name)
        mandatory_inputs = list(node.inputs.traits(mandatory=True).keys())
        other_inputs = additional_inputs[node_name] if node_name in additional_inputs else []
        for field in set(mandatory_inputs + other_inputs):
            if isdefined(getattr(node.inputs, field)):
                pass
            else:
                with pytest.raises(Exception):  # noqa: PT011
                    workflow.connect([(dummy_node, node, [('dummy', field)])])
