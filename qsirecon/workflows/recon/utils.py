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


def init_brainswipes_figures_wf(method, name="brainswipes_figures_wf"):
    """Create figures for the BrainSwipes tool.

    Parameters
    ----------
    name : str, optional
        Workflow name (default: 'brainswipes_figures_wf').

    Inputs
    ------
    anat_file : str
        Path to the anatomical image.
    dec_file : str
        Path to the DEC image.
    mask_file : str
        Path to the mask image.
    """
    from nipype.interfaces.ants.visualization import CreateTiledMosaic

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "anat_file",
                "dec_file",
                "fa_file",
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

        png_buffer = pe.MapNode(
            niu.IdentityInterface(fields=["slice_png"]),
            name="png_buffer",
            iterfield=["slice_png"],
        )

        if method == "richie":
            create_richie_mosaic = pe.MapNode(
                niu.Function(
                    input_names=["anat_file", "dec_file", "fa_file", "idx", "axis"],
                    output_names=["slice_png"],
                    function=_plot_thing,
                ),
                iterfield=["idx"],
                name=f"create_richie_mosaic_axis-{axis}",
            )
            create_richie_mosaic.inputs.axis = axis
            workflow.connect([
                (inputnode, create_richie_mosaic, [
                    ("anat_file", "anat_file"),
                    ("dec_file", "dec_file"),
                    ("fa_file", "fa_file"),
                ]),
                (loop_slice_indices, create_richie_mosaic, [("slice_idx", "idx")]),
            ])  # fmt:skip
        else:
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
                (create_tiled_mosaic, png_buffer, [("out_file", "slice_png")]),
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
            (png_buffer, create_gif, [("slice_png", "slice_png_paths")]),
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


def init_dwires_png_space_wf(res="dwi", name="dwires_png_space_wf"):
    """Create an output space with dwi voxel size that is ideal for creating square png images.

    Parameters
    ----------
    res : str
        "dwi" or "anat" - which resolution to do calculations in

    Inputs
    ------
    dwi_file : str
        Path to qsiprep-preprocessed dwi file
    mask_hires : str
        Path to the brain mask from qsiprep. If res is "anat" it should be a
        cube already
    anat_hires : str
        Path to the brain mask from qsiprep. If res is "anat" it should be a
        cube already

    Outputs
    -------
    fa_pngres : str
        Path to the fractional anisotropy image in png space
    dec_pngres : str
        Path to the DEC image in png space
    """
    from nipype.interfaces.afni.utils import Zeropad
    from nipype.interfaces.ants.resampling import ApplyTransforms

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi_file",
                "mask_hires",
                "anat_hires",
                "bval",
                "bvec",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fa_pngres", "dec_pngres"]),
        name="outputnode",
    )
    PNGRES_SIZE = 90

    buffernode = pe.Node(
        niu.IdentityInterface(fields=["dwi", "mask", "anat"]),
        name="buffernode",
    )
    if res == "dwi":
        # Zeropad the DWI so it's a cube
        zeropad_dwi = pe.Node(
            Zeropad(
                R=PNGRES_SIZE,
                L=PNGRES_SIZE,
                A=PNGRES_SIZE,
                P=PNGRES_SIZE,
                I=PNGRES_SIZE,
                S=PNGRES_SIZE,
            ),
            name="zeropad_dwi",
        )
        workflow.connect([
            (inputnode, zeropad_dwi, [("dwi_file", "in_files")]),
            (zeropad_dwi, buffernode, [("out_file", "dwi")]),
        ])  # fmt:skip

        # Get the brainmask in pngref space
        warp_mask_to_pngref = pe.Node(
            ApplyTransforms(dimension=3, interpolation="NearestNeighbor"),
            name="warp_mask_to_pngref",
        )
        workflow.connect([
            (inputnode, warp_mask_to_pngref, [("mask_hires", "input_image")]),
            (zeropad_dwi, warp_mask_to_pngref, [("out_file", "reference_image")]),
            (warp_mask_to_pngref, buffernode, [("output_image", "mask")]),
        ])  # fmt:skip

        # Resample the anat file into pngres
        warp_anat_to_pngref = pe.Node(
            ApplyTransforms(dimension=3, interpolation="NearestNeighbor"),
            name="warp_anat_to_pngref",
        )
        workflow.connect([
            (inputnode, warp_anat_to_pngref, [("anat_hires", "input_image")]),
            (zeropad_dwi, warp_anat_to_pngref, [("out_file", "reference_image")]),
            (warp_anat_to_pngref, buffernode, [("output_image", "anat")]),
        ])  # fmt:skip
    elif res == "anat":
        # Warp the DWI to the hi-res space
        warp_dwi_to_hires = pe.Node(
            ApplyTransforms(dimension=3, input_image_type=3, interpolation="BSpline"),
            name="warp_dwi_to_hires",
        )
        workflow.connect([
            (inputnode, warp_dwi_to_hires, [
                ("dwi_file", "input_image")
                ("mask_hires", "reference_image"),
            ]),
            (inputnode, buffernode, [
                ("mask_hires", "mask"),
                ("anat_hires", "anat"),
            ]),
            (warp_dwi_to_hires, buffernode, [("output_image", "dwi")]),
        ])  # fmt:skip

    # Run tensor model and get FA and DEC
    tensor_model = pe.Node(
        niu.Function(
            input_names=["dwi_file", "mask_file", "bval_file", "bvec_file"],
            output_names=["fa_file", "dec_file"],
            function=_run_tensor_model,
        ),
        name="tensor_model",
    )
    workflow.connect([
        (inputnode, tensor_model, [
            ("bval", "bval_file"),
            ("bvec", "bvec_file"),
        ]),
        (buffernode, tensor_model, [
            ("dwi", "dwi_file"),
            ("mask", "mask_file"),
        ]),
        (tensor_model, outputnode, [
            ("fa_file", "fa_pngres"),
            ("dec_file", "dec_pngres"),
        ]),
    ])  # fmt:skip

    return workflow


def init_hires_png_space_wf(name="hires_png_space_wf"):
    """Create an output space that is ideal for creating square png images.

    Parameters
    ----------

    mask_hires: str
        Path to the brain mask from qsiprep

    anat_hires: str
        Path to high-res anatomical image from qsiprep. Can be T1w or T2w


    Returns
    -------

    mask_pngres: str
        Path to the brain mask in png space

    anat_pngres: str
        Path to the anatomical image in png space

    """
    from nipype.interfaces.afni.utils import Autobox, Zeropad
    from nipype.interfaces.ants.resampling import ApplyTransforms
    from nipype.interfaces.ants.utils import ImageMath

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "mask_hires",
                "anat_hires",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["mask_pngres", "anat_pngres"]),
        name="outputnode",
    )

    autobox_mask = pe.Node(
        Autobox(),
        name="autobox_mask",
    )
    workflow.connect([(inputnode, autobox_mask, [("mask_hires", "in_file")])])

    zeropad_mask = pe.Node(
        Zeropad(R=128, L=128, A=128, P=128, I=128, S=128),
        name="zeropad_mask",
    )
    workflow.connect([
        (autobox_mask, zeropad_mask, [("out_file", "in_files")]),
        (zeropad_mask, outputnode, [("out_file", "mask_pngres")]),
    ])  # fmt:skip

    warp_anat_to_pngref = pe.Node(
        ApplyTransforms(dimension=3, interpolation="NearestNeighbor"),
        name="warp_anat_to_pngref",
    )
    workflow.connect([
        (inputnode, warp_anat_to_pngref, [("anat_hires", "input_image")]),
        (zeropad_mask, warp_anat_to_pngref, [("out_file", "reference_image")]),
    ])  # fmt:skip

    scale_anat = pe.Node(
        ImageMath(
            operation="TruncateImageIntensity",
            op2="0.02 0.98 256",
        ),
        name="scale_anat",
    )
    workflow.connect([
        (warp_anat_to_pngref, scale_anat, [("output_image", "op1")]),
        (scale_anat, outputnode, [("output_image", "anat_pngres")]),
    ])  # fmt:skip

    return workflow


def _run_tensor_model(dwi_file, mask_file, bval_file, bvec_file):
    import os

    import dipy.reconst.dti as dti
    import numpy as np
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.io.image import load_nifti, save_nifti

    fa_file = os.path.abspath("fa.nii.gz")
    dec_file = os.path.abspath("dec.nii.gz")

    # Use DIPY to fit a tensor
    data, affine = load_nifti(dwi_file)
    mask_data, _ = load_nifti(mask_file)
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask_data > 0)

    # Get FA and DEC from the tensor fit
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA = np.clip(FA, 0, 1)

    # Convert to colorFA image as in DIPY documentation
    FA_masked = FA * mask_data
    RGB = dti.color_fa(FA_masked, tenfit.evecs)
    RGB = np.array(255 * RGB, "uint8")
    save_nifti(fa_file, FA_masked.astype(np.float32), affine)
    save_nifti(dec_file, RGB, affine)
    return fa_file, dec_file


def get_anat_rgba_slices(anat_data, rgb_data, fa_data, idx, axis):
    import numpy as np

    # Select slice from axis and handle rotation
    if axis == 0:
        anat = anat_data[idx, :, :]
        rgb = rgb_data[idx, :, :]
        fa = fa_data[idx, :, :]
    elif axis == 1:
        anat = anat_data[:, idx, :]
        rgb = rgb_data[:, idx, :]
        fa = fa_data[:, idx, :]
    else:
        anat = anat_data[:, :, idx]
        rgb = rgb_data[:, :, idx]
        fa = fa_data[:, :, idx]
    rgba = np.concatenate([rgb, fa[:, :, np.newaxis]], axis=-1)
    return anat, rgba


def _plot_thing(anat, rgb, fa, slice_idx, axis):
    from qsirecon.workflows.recon.utils import get_anat_rgba_slices

    # Make the gifs!
    temp_image_files = []
    for axis in [0, 1, 2]:
        # Compute the indices of the slices
        slice_indices = get_anchor_slices_from_mask(mask_file, axis)
        axis_slice_names = slice_names[axis]
        named_slices = zip(slice_indices, axis_slice_names)

        for base_slice_idx, slice_name in named_slices:
            output_gif_path = f"{prefix}{slice_name}.gif"
            images = []

            for offset_idx, slice_offset in enumerate(slice_gif_offsets):
                slice_idx = base_slice_idx + slice_offset
                slice_png_path = f"{prefix}{slice_name}_part-{offset_idx}_dec.png"
                fig, ax = plt.subplots(1, 1, figsize=my_figsize)
                slice_anat, slice_rgba = get_anat_rgba_slices(slice_idx, axis)

                _ = ax.imshow(
                    slice_anat,
                    vmin=anat_vmin,
                    vmax=anat_vmax,
                    cmap=plt.cm.Greys_r,
                )
                _ = ax.imshow(slice_rgba.astype(np.uint8))
                _ = ax.axis("off")

                fig.savefig(slice_png_path, bbox_inches="tight")
                plt.close(fig)
                images.append(load_and_rotate_png(slice_png_path, axis))
                temp_image_files.append(slice_png_path)

            # Create a back and forth animation by appending the images to
            # themselves, but reversed
            images = images + images[-2:0:-1]

            # Save the gif
            imageio.mimsave(
                output_gif_path,
                images,
                loop=0,
                duration=(1 / fps) * 1000,
                subrectangles=True,
            )

def richie_fa_gifs(dec_file, fa_file, anat_file, mask_file, prefix):
    """Create GIFs for Swipes using ARH's method from HBN-POD2.

    Parameters
    ----------

    dec_file: str
        Path to an RGB DEC NIFTI file resampled into pngres space

    fa_file: str
        Path to a NIFTI file of FA values in pngres space

    anat_file: str
        Path to anatomical file to display in grayscale behind the RBGA data.
        Also must be in pngres space

    mask_file: str
        Path to the brainmask NIFTI file in pngres space

    prefix: str
        Stem of the gifs that will be written

    Returns: None

    """

    anat_img = nb.load(anat_file)
    anat_data = anat_img.get_fdata()
    anat_vmin, anat_vmax = scoreatpercentile(anat_data.flatten(), [1, 98])

    # Load the RGB data. It was created by tortoise, but resampled into
    # pngres space. This also converts it to a 3-vector data type.
    rgb_img = nb.load(dec_file)
    rgb_data = rgb_img.get_fdata().squeeze()
    rgb_data = np.clip(0, 255, rgb_data * BRIGHTNESS_UPSCALE)

    # Open FA image and turn it into alpha values
    fa_img = nb.load(fa_file)
    fa_data = fa_to_alpha(np.clip(0, 1, fa_img.get_fdata())) * 255

    print(f"Setting grayscale vmax to {anat_vmax}")

    def get_anat_rgba_slices(idx, axis):
        # Select slice from axis and handle rotation
        if axis == 0:
            anat = anat_data[idx, :, :]
            rgb = rgb_data[idx, :, :]
            fa = fa_data[idx, :, :]
        elif axis == 1:
            anat = anat_data[:, idx, :]
            rgb = rgb_data[:, idx, :]
            fa = fa_data[:, idx, :]
        else:
            anat = anat_data[:, :, idx]
            rgb = rgb_data[:, :, idx]
            fa = fa_data[:, :, idx]
        rgba = np.concatenate([rgb, fa[:, :, np.newaxis]], axis=-1)
        return anat, rgba

    # Make the gifs!
    temp_image_files = []
    for axis in [0, 1, 2]:
        # Compute the indices of the slices
        slice_indices = get_anchor_slices_from_mask(mask_file, axis)
        axis_slice_names = slice_names[axis]
        named_slices = zip(slice_indices, axis_slice_names)

        for base_slice_idx, slice_name in named_slices:
            output_gif_path = f"{prefix}{slice_name}.gif"
            images = []

            for offset_idx, slice_offset in enumerate(slice_gif_offsets):
                slice_idx = base_slice_idx + slice_offset
                slice_png_path = f"{prefix}{slice_name}_part-{offset_idx}_dec.png"
                fig, ax = plt.subplots(1, 1, figsize=my_figsize)
                slice_anat, slice_rgba = get_anat_rgba_slices(slice_idx, axis)

                _ = ax.imshow(
                    slice_anat,
                    vmin=anat_vmin,
                    vmax=anat_vmax,
                    cmap=plt.cm.Greys_r,
                )
                _ = ax.imshow(slice_rgba.astype(np.uint8))
                _ = ax.axis("off")

                fig.savefig(slice_png_path, bbox_inches="tight")
                plt.close(fig)
                images.append(load_and_rotate_png(slice_png_path, axis))
                temp_image_files.append(slice_png_path)

            # Create a back and forth animation by appending the images to
            # themselves, but reversed
            images = images + images[-2:0:-1]

            # Save the gif
            imageio.mimsave(
                output_gif_path,
                images,
                loop=0,
                duration=(1 / fps) * 1000,
                subrectangles=True,
            )
