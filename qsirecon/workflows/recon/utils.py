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
from ...interfaces.recon_scalars import OrganizeScalarData
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
        niu.IdentityInterface(fields=["scalar_files"]),
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
        iterfield=["in_file", "meta_dict", "model", "param"],
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
        ]),
        (ds_scalar, outputnode, [("out_file", "scalar_files")]),
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
        "Testing Workflow\n\n: This workflow tests boilerplate, figures and derivatives"
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


def init_brainswipes_figures_wf(name="brainswipes_figures_wf"):
    """Create figures for the BrainSwipes tool."""
    from nipype.interfaces.ants.visualization import CreateTiledMosaic

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "anat_file",
                "dec_file",
                "mask_file",
            ],
        ),
        name="inputnode",
    )

    workflow = Workflow(name=name)

    slice_names = [
        [
            "LeftTemporalSagittal",
            "LeftBrainStemSagittal",
            "RightTemporalSagittal",
        ],
        [
            "AnteriorCoronal",
            "PosteriorCoronal",
        ],
        [
            "CerebellumAxial",
            "SemiovaleAxial",
        ],
    ]

    # Loop is for individual slices in the gif image
    for axis in [0, 1, 2]:
        axis_slice_names = slice_names[axis]

        # Compute the indices of the slices
        get_anchor_slices = pe.Node(
            niu.Function(
                input_names=["mask_file", "axis"],
                output_names=["slice_indices"],
                function=_get_anchor_slices_from_mask,
            ),
            name=f"get_anchor_slices_axis-{axis}",
        )
        get_anchor_slices.inputs.axis = axis
        workflow.connect([(inputnode, get_anchor_slices, [("mask_file", "mask_file")])])

        loop_slice_indices = pe.MapNode(
            niu.Function(
                input_names=["base_slice_idx"],
                output_names=["slice_idx", "duration"],
                function=_loop_slice_indices,
            ),
            iterfield=["base_slice_idx"],
            name=f"loop_slice_indices_axis-{axis}",
        )
        workflow.connect([
            (get_anchor_slices, loop_slice_indices, [("slice_indices", "base_slice_idx")]),
        ])  # fmt:skip

        format_slice_indices = pe.MapNode(
            niu.Function(
                input_names=["slice_indices"],
                output_names=["slice_idx"],
                function=_format_slice_index,
            ),
            iterfield=["slice_indices"],
            name=f"format_slice_indices_axis-{axis}",
        )
        workflow.connect([
            (loop_slice_indices, format_slice_indices, [("slice_idx", "slice_indices")]),
        ])  # fmt:skip

        kwargs = {}
        if axis in [0, 1]:
            kwargs["flip_slice"] = "0x1"
        create_tiled_mosaic = pe.MapNode(
            CreateTiledMosaic(alpha_value=0.65, tile_geometry="1x1", direction=axis, **kwargs),
            iterfield=["slice_idx", "slice_name"],
        )
        create_tiled_mosaic.inputs.slice_name = axis_slice_names
        workflow.connect([
            (inputnode, create_tiled_mosaic, [
                ("anat_file", "input_image"),
                ("dec_file", "rgb_image"),
                ("mask_file", "mask_image"),
            ]),
            (format_slice_indices, create_tiled_mosaic, [("slice_idx", "slice_idx")]),
        ])  # fmt:skip

        create_gif = pe.Node(
            niu.Function(
                inputnames=["slice_png_paths", "duration"],
                outputnames=["output_gif_path"],
                function=_create_gif,
            ),
            name=f"create_gif_axis-{axis}",
        )
        workflow.connect([
            (loop_slice_indices, create_gif, [("duration", "duration")]),
            (create_tiled_mosaic, create_gif, [("output_image", "slice_png_paths")]),
        ])  # fmt:skip

        ds_gif = pe.MapNode(
            DerivativesDataSink(),
            name=f"ds_gif_axis-{axis}",
            iterfield=["in_file", "desc"],
        )
        ds_gif.inputs.desc = axis_slice_names
        workflow.connect([(create_gif, ds_gif, [("output_gif_path", "in_file")])])

    return workflow


def _get_anchor_slices_from_mask(mask_file, axis):
    """Find the slice numbers for ``slice_ratios`` inside a mask."""
    import nibabel as nb
    import numpy as np

    # Specify the slices that we would like to save as a fraction
    # of the masked extent.
    slice_ratios = [
        np.array([0.2, 0.48, 0.8]),  # LR Slices
        np.array([0.4, 0.6]),  # AP Slices
        np.array([0.2, 0.7]),  # IS Slices
    ]

    mask_arr = nb.load(mask_file).get_fdata()
    mask_coords = np.argwhere(mask_arr > 0)[:, axis]
    min_slice = mask_coords.min()
    max_slice = mask_coords.max()
    covered_slices = max_slice - min_slice
    return np.floor((covered_slices * slice_ratios[axis]) + min_slice).astype(np.int64)


def _loop_slice_indices(base_slice_idx):
    import numpy as np

    # Specify that slice range for the animated gifs
    slice_gif_offsets = np.arange(-5, 6)
    # Calculate the frames-per-second for the animated gifs
    fps = len(slice_gif_offsets) / 2.0

    duration = (1 / fps) * 1000

    return base_slice_idx + np.arange(-5, 6), duration


def _format_slice_index(idx):
    return f"{idx}x{idx}"


def _create_gif(slice_png_paths, duration, output_gif_path=None):
    import os

    import imageio
    from PIL import Image

    if not output_gif_path:
        output_gif_path = os.path.abspath("animated.gif")

    images = []
    for slice_png_path in slice_png_paths:
        # Upsample the image ONCE
        ants_png = Image.open(slice_png_path)
        resized = ants_png.resize((512, 512), Image.NEAREST)
        images.append(resized)

    # Create a back and forth animation by appending the images to
    # themselves, but reversed
    images = images + images[-2:0:-1]

    # Save the gif
    imageio.mimsave(
        output_gif_path,
        images,
        loop=0,
        duration=duration,
        subrectangles=True,
    )

    return output_gif_path
