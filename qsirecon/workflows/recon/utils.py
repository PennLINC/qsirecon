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


def init_dwires_png_space_wf(res="dwi", name="dwires_png_space_wf"):
    """Create an output space with dwi voxel size that is ideal for
    creating square png images.

    Parameters
    ----------
    res : str
        "dwi" or "anat" - which resolution to do calculations in

    Inputs
    ------
    dwi_file: str
        Path to qsiprep-preprocessed dwi file
    hires_maskfile: str
        Path to the brain mask from qsiprep. If res is "anat" it should be a
        cube already
    hires_anatfile: str
        Path to the brain mask from qsiprep. If res is "anat" it should be a
        cube already

    Outputs
    -------
    pngres_dec
        PNG-resolution DEC image
    pngres_fa
        PNG-resolution FA image
    pngres_mask : str
        Path to the brain mask in png space
    pngres_anat : str
        Path to the anatomical image in png space
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi_file",
                "hires_maskfile",
                "hires_anatfile",
                "res",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["pngres_maskfile", "pngres_anatfile"]),
        name="outputnode",
    )
    PNGRES_SIZE = 90

    # Create a cube bounding box that we will use to take pics
    pngres_dwi = op.abspath(f"{res}pngres_dwi.nii")
    pngres_anat = op.abspath(f"{res}pngres_anat.nii")
    pngres_dec = op.abspath(f"{res}pngres_dec.nii")
    pngres_fa = op.abspath(f"{res}pngres_fa.nii")
    pngres_mask = op.abspath(f"{res}pngres_mask.nii")
    fstem = dwi_file.replace(".nii.gz", "")

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
        workflow.connect([(inputnode, zeropad_dwi, [("dwi_file", "in_files")])])

        # Get the brainmask in pngref space
        warp_mask_to_pngref = pe.Node(
            ApplyTransforms(dimension=3, interpolation="NearestNeighbor"),
            name="warp_mask_to_pngref",
        )
        workflow.connect([
            (inputnode, warp_mask_to_pngref, [("hires_maskfile", "input_image")]),
            (zeropad_dwi, warp_mask_to_pngref, [("out_file", "reference_image")]),
        ])  # fmt:skip

        # Resample the anat file into pngres
        warp_anat_to_pngref = pe.Node(
            ApplyTransforms(dimension=3, interpolation="NearestNeighbor"),
            name="warp_anat_to_pngref",
        )
        workflow.connect([
            (inputnode, warp_anat_to_pngref, [("hires_anatfile", "input_image")]),
            (zeropad_dwi, warp_anat_to_pngref, [("out_file", "reference_image")]),
        ])  # fmt:skip
    elif res == "anat":
        subprocess.run(
            [
                "antsApplyTransforms",
                "-d",
                "3",
                "-e",
                "3",
                "-i",
                dwi_file,
                "-r",
                hires_maskfile,
                "-o",
                pngres_dwi,
                "-v",
                "1",
                "--interpolation",
                "BSpline",
            ],
            check=True,
        )
        pngres_mask = hires_maskfile
        pngres_anat = hires_anatfile

    # Use DIPY to fit a tensor
    data, affine = load_nifti(pngres_dwi)
    mask_data, _ = load_nifti(pngres_mask)
    bvals, bvecs = read_bvals_bvecs(f"{fstem}.bval", f"{fstem}.bvec")
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    print(f"Fitting Tensor to {pngres_dwi}")
    tenfit = tenmodel.fit(data, mask=mask_data > 0)

    # Get FA and DEC from the tensor fit
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA = np.clip(FA, 0, 1)

    # Convert to colorFA image as in DIPY documentation
    FA_masked = FA * mask_data
    RGB = dti.color_fa(FA_masked, tenfit.evecs)
    RGB = np.array(255 * RGB, 'uint8')
    save_nifti(pngres_fa, FA_masked.astype(np.float32), affine)
    save_nifti(pngres_dec, RGB, affine)

    return workflow


def init_hires_png_space_wf(name="hires_png_space_wf"):
    """Create an output space that is ideal for creating square png images.

    Parameters
    ----------

    hires_maskfile: str
        Path to the brain mask from qsiprep

    hires_anatfile: str
        Path to high-res anatomical image from qsiprep. Can be T1w or T2w


    Returns
    -------

    pngres_maskfile: str
        Path to the brain mask in png space

    pngres_anatfile: str
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
