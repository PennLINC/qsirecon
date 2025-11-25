"""
Miscellaneous workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_discard_repeated_samples_wf
.. autofunction:: init_conform_dwi_wf


"""

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ...interfaces import ConformDwi
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.gradients import GradientSelect, RemoveDuplicates
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.mrtrix import MRTrixGradientTable
from ...interfaces.recon_scalars import DisorganizeScalarData, OrganizeScalarData
from ...interfaces.utils import TestReportPlot, WriteSidecar
from ...utils.bids import clean_datasinks


def init_conform_dwi_wf(inputs_dict, name="conform_dwi", qsirecon_suffix="", params={}):
    """If data were preprocessed elsewhere, ensure the gradients and images
    conform to LPS+ before running other parts of the pipeline."""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["dwi_file", "bval_file", "bvec_file", "b_file"]),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    workflow.__desc__ = "The dMRI data were conformed to LPS+ orientation."

    conform = pe.Node(ConformDwi(), name="conform_dwi")
    grad_table = pe.Node(MRTrixGradientTable(), name="grad_table")
    workflow.connect([
        (inputnode, conform, [
            ('dwi_file', 'dwi_file')]),
        (conform, grad_table, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (grad_table, outputnode, [
            ('gradient_file', 'b_file')]),
        (conform, outputnode, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_file', 'dwi_file')])
    ])  # fmt:skip
    return workflow


def init_discard_repeated_samples_wf(
    inputs_dict,
    name="discard_repeats",
    qsirecon_suffix="",
    params={},
):
    """Remove a sample if a similar direction/gradient has already been sampled."""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi_file",
                "bval_file",
                "bvec_file",
                "local_bvec_file",
                "b_file",
                "btable_file",
            ]
        ),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    workflow.__desc__ = (
        "Volumes in the dMRI data were discarded if a similar direction/gradient has already been "
        "sampled. "
        "A volume was classified as a duplicate if the distance between its scaled gradient "
        f"vector and a previous volume's was less than {params.get('distance_cutoff', 5.0)} "
        "s / mm^2."
    )

    discard_repeats = pe.Node(RemoveDuplicates(**params), name="discard_repeats")
    workflow.connect([
        (inputnode, discard_repeats, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('b_file', 'b_file'),
            ('btable_file', 'btable_file'),
            ('bvec_file', 'bvec_file')]),
        (discard_repeats, outputnode, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('b_file', 'b_file'),
            ('btable_file', 'btable_file'),
            ('bvec_file', 'bvec_file')])
    ])  # fmt:skip

    return workflow


def init_gradient_select_wf(
    inputs_dict,
    name="gradient_select_wf",
    qsirecon_suffix="",
    params={},
):
    """Remove a sample if a similar direction/gradient has already been sampled."""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi_file",
                "bval_file",
                "bvec_file",
                "local_bvec_file",
                "b_file",
                "btable_file",
            ]
        ),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    workflow.__desc__ = "Gradients were selected based on the requested shells."

    gradient_select = pe.Node(GradientSelect(**params), name="gradient_select")
    workflow.connect([
        (inputnode, gradient_select, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('b_file', 'b_file'),
            ('btable_file', 'btable_file'),
            ('bvec_file', 'bvec_file')]),
        (gradient_select, outputnode, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('b_file', 'b_file'),
            ('btable_file', 'btable_file'),
            ('bvec_file', 'bvec_file')])
    ])  # fmt:skip

    return workflow


def init_scalar_output_wf(
    name="scalar_output_wf",
):
    """Write out reconstructed scalar maps."""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "source_file",
                "scalar_configs",
                # Entities
                "space",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["scalar_files", "scalar_configs"]),
        name="outputnode",
    )
    workflow = Workflow(name=name)

    organize_scalar_data = pe.MapNode(
        OrganizeScalarData(),
        iterfield=["scalar_config"],
        name="organize_scalar_data",
    )
    workflow.connect([(inputnode, organize_scalar_data, [("scalar_configs", "scalar_config")])])

    ds_scalar = pe.MapNode(
        DerivativesDataSink(
            dismiss_entities=["desc"],
            datatype="dwi",
            suffix="dwimap",
            extension="nii.gz",
        ),
        iterfield=["in_file", "meta_dict", "model", "param", "desc"],
        name="ds_scalar",
        run_without_submitting=True,
    )
    workflow.connect([
        (inputnode, ds_scalar, [
            ("source_file", "source_file"),
            ("space", "space"),
        ]),
        (organize_scalar_data, ds_scalar, [
            ("scalar_file", "in_file"),
            ("metadata", "meta_dict"),
            ("model", "model"),
            ("param", "param"),
            ("desc", "desc"),
        ]),
        (ds_scalar, outputnode, [("out_file", "scalar_files")]),
    ])  # fmt:skip

    disorganize_scalar_data = pe.MapNode(
        DisorganizeScalarData(),
        iterfield=["scalar_config", "scalar_file"],
        name="disorganize_scalar_data",
    )
    workflow.connect([
        (inputnode, disorganize_scalar_data, [("scalar_configs", "scalar_config")]),
        (ds_scalar, disorganize_scalar_data, [("out_file", "scalar_file")]),
        (disorganize_scalar_data, outputnode, [("scalar_config", "scalar_configs")]),
    ])  # fmt:skip

    return workflow


def init_test_wf(inputs_dict, name="test_wf", qsirecon_suffix="test", params={}):
    """A workflow for testing how derivatives will be saved."""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fibgz", "recon_scalars"]), name="outputnode"
    )
    workflow = Workflow(name=name)
    outputnode.inputs.recon_scalars = []
    workflow.__desc__ = (
        "\n\n#### Testing Workflow\n\nThis workflow tests boilerplate, figures and derivatives."
    )

    write_metadata = pe.Node(WriteSidecar(metadata=inputs_dict), name="write_metadata")
    plot_image = pe.Node(TestReportPlot(), name="plot_image")

    ds_metadata = pe.Node(
        DerivativesDataSink(desc="availablemetadata"),
        name="ds_metadata",
        run_without_submitting=True,
    )
    ds_plot = pe.Node(
        DerivativesDataSink(desc="exampleplot", datatype="figures", extension=".png"),
        name="ds_plot",
        run_without_submitting=True,
    )
    workflow.connect([
        (inputnode, plot_image, [("dwi_file", "dwi_file")]),
        (write_metadata, ds_metadata, [("out_file", "in_file")]),
        (plot_image, ds_plot, [("out_file", "in_file")]),
    ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)
