#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from pathlib import Path

import nipype.interfaces.io as nio
from nipype.interfaces import afni, ants, mrtrix3
from nipype.interfaces import utility as niu
from nipype.interfaces.base import traits
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from ... import config
from ...interfaces.anatomical import GetTemplate, VoxelSizeChooser
from ...interfaces.ants import ConvertTransformFile
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.freesurfer import find_fs_path
from ...interfaces.gradients import ExtractB0s
from ...interfaces.interchange import (
    FS_FILES_TO_REGISTER,
    anatomical_workflow_outputs,
    qsiprep_highres_anatomical_ingressed_fields,
    recon_workflow_input_fields,
)
from ...interfaces.mrtrix import GenerateMasked5tt, ITKTransformConvert, TransformHeader
from ...interfaces.utils import WarpConnectivityAtlases
from ...utils.bids import clean_datasinks
from ...utils.boilerplate import describe_atlases

# Required freesurfer files for mrtrix's HSV 5tt generation
HSV_REQUIREMENTS = [
    "mri/aparc+aseg.mgz",
    "mri/brainmask.mgz",
    "mri/norm.mgz",
    "mri/transforms/talairach.xfm",
    "surf/lh.white",
    "surf/lh.pial",
    "surf/rh.white",
    "surf/rh.pial",
]


def init_highres_recon_anatomical_wf(
    subject_id,
    extras_to_make,
    status,
):
    """Gather any high-res anatomical data (images, transforms, segmentations) to use
    in recon workflows.

    This workflow searches through input data to see what anatomical data is available.
    The anatomical data may be in a freesurfer directory.
    """
    workflow = Workflow(name="recon_anatomical_wf")

    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=anatomical_workflow_outputs),
        name="outputnode",
    )
    workflow.__desc__ = ""
    qsirecon_suffix = ""

    # If there is no high-res anat data in the inputs there may still be an image available
    # from freesurfer. Check for it:
    freesurfer_dir = config.execution.freesurfer_input
    subject_freesurfer_path = find_fs_path(freesurfer_dir, subject_id)
    status["has_freesurfer"] = subject_freesurfer_path is not None
    status["has_qsiprep_5tt_hsvs"] = False
    status["has_freesurfer_5tt_hsvs"] = False

    # If no high-res are available, we're done here
    if not status["has_qsiprep_t1w"] and not status["has_freesurfer"]:
        config.loggers.workflow.warning(
            f"No high-res anatomical data available directly in recon inputs for {subject_id}."
        )
        # If a 5tt image is needed, this is an error
        if "mrtrix_5tt_hsvs" in extras_to_make:
            raise Exception("FreeSurfer data is required to make HSVS 5tt image.")

        workflow.add_nodes([outputnode])
        workflow = clean_datasinks(workflow, qsirecon_suffix)
        return workflow, status

    config.loggers.workflow.info(
        f"Found high-res anatomical data in preprocessed inputs for {subject_id}."
    )
    workflow.connect([
        (inputnode, outputnode, [
            (name, name) for name in qsiprep_highres_anatomical_ingressed_fields
        ]),
    ])  # fmt:skip

    # grab un-coregistered freesurfer images for later use
    if status["has_freesurfer"]:
        config.loggers.workflow.info(
            f"Freesurfer directory {subject_freesurfer_path} exists for {subject_id}"
        )
        fs_source = pe.Node(
            nio.FreeSurferSource(
                subjects_dir=str(subject_freesurfer_path.parent),
                subject_id=subject_freesurfer_path.name,
            ),
            name="fs_source",
        )

    # Do we need to calculate anything else on the fly
    if "mrtrix_5tt_hsvs" in extras_to_make:
        # Check for specific files needed for Hybrid Surface and Volume Segmentation (HSVS).
        missing_fs_hsvs_files = check_hsv_inputs(Path(subject_freesurfer_path))
        if missing_fs_hsvs_files:
            raise Exception(" ".join(missing_fs_hsvs_files) + "are missing: unable to make a HSV.")

        config.loggers.workflow.info("FreeSurfer data will be used to create a HSVS 5tt image.")
        status["has_freesurfer_5tt_hsvs"] = True
        create_5tt_hsvs = pe.Node(
            GenerateMasked5tt(
                algorithm="hsvs",
                in_file=str(subject_freesurfer_path),
                nthreads=config.nipype.omp_nthreads,
            ),
            name="create_5tt_hsvs",
            n_procs=config.nipype.omp_nthreads,
        )
        ds_qsiprep_5tt_hsvs = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                atlas="hsvs",
                space="T1w",
                suffix="dseg",
            ),
            name="ds_qsiprep_5tt_hsvs",
            run_without_submitting=True,
        )
        ds_fs_5tt_hsvs = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                desc="hsvs",
                space="fsnative",
                suffix="dseg",
                compress=True,
            ),
            name="ds_fs_5tt_hsvs",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, ds_fs_5tt_hsvs, [("acpc_preproc", "source_file")]),
            (inputnode, ds_qsiprep_5tt_hsvs, [("acpc_preproc", "source_file")]),
            (create_5tt_hsvs, outputnode, [("out_file", "fs_5tt_hsvs")]),
            (create_5tt_hsvs, ds_fs_5tt_hsvs, [("out_file", "in_file")]),
        ])  # fmt:skip

        # Transform the 5tt image so it's registered to the QSIRecon AC-PC T1w
        if status["has_qsiprep_t1w"]:
            config.loggers.workflow.info(
                "HSVS 5tt imaged will be registered to the QSIRecon T1w image."
            )
            status["has_qsiprep_5tt_hsvs"] = True
            register_fs_to_qsiprep_wf = init_register_fs_to_qsiprep_wf(
                use_qsiprep_reference_mask=True,
            )
            apply_header_to_5tt = pe.Node(TransformHeader(), name="apply_header_to_5tt")
            workflow.connect([
                (inputnode, register_fs_to_qsiprep_wf, [
                    ("acpc_preproc", "inputnode.qsiprep_reference_image"),
                    ("acpc_brain_mask", "inputnode.qsiprep_reference_mask"),
                ]),
                (fs_source, register_fs_to_qsiprep_wf, [
                    (field, f"inputnode.{field}") for field in FS_FILES_TO_REGISTER
                ]),
                (register_fs_to_qsiprep_wf, outputnode, [
                    ("outputnode.fs_to_qsiprep_transform_mrtrix",
                        "fs_to_qsiprep_transform_mrtrix"),
                    ("outputnode.fs_to_qsiprep_transform_itk", "fs_to_qsiprep_transform_itk")
                ] + [
                    (f"outputnode.{field}", field) for field in FS_FILES_TO_REGISTER
                ]),
                (create_5tt_hsvs, apply_header_to_5tt, [("out_file", "in_image")]),
                (register_fs_to_qsiprep_wf, apply_header_to_5tt, [
                    ("outputnode.fs_to_qsiprep_transform_mrtrix", "transform_file"),
                ]),
                (apply_header_to_5tt, outputnode, [("out_image", "qsiprep_5tt_hsvs")]),
                (apply_header_to_5tt, ds_qsiprep_5tt_hsvs, [("out_image", "in_file")]),
            ])  # fmt:skip

            workflow.__desc__ += "A hybrid surface/volume segmentation was created [Smith 2020]."

    workflow = clean_datasinks(workflow, qsirecon_suffix)
    return workflow, status


def check_hsv_inputs(subj_fs_path):
    """Determine if a FreeSurfer directory has the required files for HSV."""
    missing = []
    for requirement in HSV_REQUIREMENTS:
        if not (subj_fs_path / requirement).exists():
            missing.append(requirement)
    return missing


def _check_zipped_unzipped(path_to_check):
    """Check to see if a path exists and warn if it's gzipped."""

    exists = False
    if path_to_check.exists():
        exists = True
    if path_to_check.name.endswith(".gz"):
        nonzipped = str(path_to_check)[:-3]
        if Path(nonzipped).exists():
            config.loggers.workflow.warn(
                "A Non-gzipped input nifti file was found. Consider gzipping %s", nonzipped
            )
            exists = True
    config.loggers.workflow.info(f"CHECKING {path_to_check}: {exists}")
    return exists


def init_register_fs_to_qsiprep_wf(
    use_qsiprep_reference_mask=False,
    name="register_fs_to_qsiprep_wf",
):
    """Registers a T1w images from freesurfer to another image and transforms"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=FS_FILES_TO_REGISTER + ["qsiprep_reference_image", "qsiprep_reference_mask"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=FS_FILES_TO_REGISTER
            + ["fs_to_qsiprep_transform_itk", "fs_to_qsiprep_transform_mrtrix"]
        ),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    workflow.__desc__ = "FreeSurfer outputs were registered to the QSIRecon outputs."

    # Convert the freesurfer inputs so we can register them with ANTs
    convert_fs_brain = pe.Node(
        mrtrix3.MRConvert(out_file="fs_brain.nii", args="-strides -1,-2,3"),
        name="convert_fs_brain",
    )

    # Register the brain to the QSIRecon reference
    ants_settings = pkgrf("qsirecon", "data/freesurfer_to_qsiprep.json")
    register_to_qsiprep = pe.Node(
        ants.Registration(from_file=ants_settings), name="register_to_qsiprep"
    )

    # If there is a mask for the QSIRecon reference image, use it
    if use_qsiprep_reference_mask:
        workflow.connect(inputnode, "qsiprep_reference_mask",
                         register_to_qsiprep, "fixed_image_masks")  # fmt:skip

    # The more recent ANTs mat format isn't compatible with transformconvert.
    # So convert it to ANTs text format with ConvertTransform
    convert_ants_transform = pe.Node(
        ConvertTransformFile(dimension=3), name="convert_ants_transform"
    )

    # Convert from ANTs text format to MRtrix3 format
    convert_ants_to_mrtrix_transform = pe.Node(
        ITKTransformConvert(), name="convert_ants_to_mrtrix_transform"
    )

    # Adjust the headers of all the input images so they're aligned to the qsiprep ref
    transform_nodes = {}
    for image_name in FS_FILES_TO_REGISTER:
        transform_nodes[image_name] = pe.Node(TransformHeader(), name="transform_" + image_name)
        workflow.connect([
            (inputnode, transform_nodes[image_name], [(image_name, "in_image")]),
            (convert_ants_to_mrtrix_transform,
             transform_nodes[image_name], [("out_transform", "transform_file")]),
            (transform_nodes[image_name], outputnode, [("out_image", image_name)])
        ])  # fmt:skip

    workflow.connect([
        (inputnode, convert_fs_brain, [("brain", "in_file")]),
        (inputnode, register_to_qsiprep, [("qsiprep_reference_image", "fixed_image")]),
        (convert_fs_brain, register_to_qsiprep, [("out_file", "moving_image")]),
        (register_to_qsiprep, convert_ants_transform, [
            (("forward_transforms", _get_first), "in_transform"),
        ]),
        (register_to_qsiprep, outputnode, [
            ("composite_transform", "fs_to_qsiprep_transform_itk"),
        ]),
        (convert_ants_transform, convert_ants_to_mrtrix_transform, [
            ("out_transform", "in_transform"),
        ]),
        (convert_ants_to_mrtrix_transform, outputnode, [
            ("out_transform", "fs_to_qsiprep_transform_mrtrix"),
        ]),
    ])  # fmt:skip

    return workflow


def init_dwi_recon_anatomical_workflow(
    atlas_configs,
    has_qsiprep_5tt_hsvs,
    needs_t1w_transform,
    has_freesurfer_5tt_hsvs,
    has_qsiprep_t1w,
    has_qsiprep_t1w_transforms,
    has_freesurfer,
    extras_to_make,
    name,
    prefer_dwi_mask=False,
):
    """Ensure that anatomical data is available for the reconstruction workflows.

    This workflow calculates images/transforms that require a DWI spatial reference.
    Specifically, three additional features are added:

      * ``"dwi_mask"``: a brain mask in the voxel space of the DWI
      * ``"atlas_configs"``: A dictionary used by connectivity workflows to get
        brain parcellations.
      * ``"odf_rois"``: An image with some interesting ROIs for plotting ODFs

    Parameters:
    ===========
        has_qsiprep_5tt_hsvs:
        has_freesurfer_5tt_hsvs: True,
        has_qsiprep_t1w:
        has_qsiprep_t1w_transforms: True}
    """
    # Inputnode holds data from the T1w-based anatomical workflow
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name="inputnode",
    )
    connect_from_inputnode = set(recon_workflow_input_fields)
    # Buffer to hold the anatomical files that are calculated here
    buffernode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name="buffernode",
    )
    connect_from_buffernode = set()
    b0_threshold = config.workflow.b0_threshold
    qsirecon_suffix = ""

    def _get_source_node(fieldname):
        if fieldname in connect_from_inputnode:
            return inputnode
        if fieldname in connect_from_buffernode:
            return buffernode
        raise Exception(f"Can't determine location of {fieldname}")

    def _exchange_fields(fields):
        connect_from_inputnode.difference_update(fields)
        connect_from_buffernode.update(fields)

    # These are always created here
    _exchange_fields(["dwi_mask", "atlas_configs", "odf_rois", "resampling_template"])

    outputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    skull_strip_method = "antsBrainExtraction"
    desc = """

#### Anatomical data for DWI reconstruction

"""

    def _get_status():
        return {
            "has_qsiprep_5tt_hsvs": has_qsiprep_5tt_hsvs,
            "has_freesurfer_5tt_hsvs": has_freesurfer_5tt_hsvs,
            "has_qsiprep_t1w": has_qsiprep_t1w,
            "has_qsiprep_t1w_transforms": has_qsiprep_t1w_transforms,
        }

    # XXX: This is a temporary solution until QSIRecon supports flexible output spaces.
    get_template = pe.Node(
        GetTemplate(
            template_name="MNI152NLin2009cAsym" if not config.workflow.infant else "MNIInfant",
        ),
        name="get_template",
    )
    mask_template = pe.Node(
        afni.Calc(expr="a*b", outputtype="NIFTI_GZ"),
        name="mask_template",
    )
    reorient_to_lps = pe.Node(
        afni.Resample(orientation="RAI", outputtype="NIFTI_GZ"),
        name="reorient_to_lps",
    )

    reference_grid_wf = init_output_grid_wf()
    workflow.connect([
        (get_template, mask_template, [
            ("template_file", "in_file_a"),
            ("mask_file", "in_file_b"),
        ]),
        (mask_template, reorient_to_lps, [("out_file", "in_file")]),
        (inputnode, reference_grid_wf, [("dwi_ref", "inputnode.input_image")]),
        (reorient_to_lps, reference_grid_wf, [("out_file", "inputnode.template_image")]),
        (reference_grid_wf, buffernode, [("outputnode.grid_image", "resampling_template")]),
    ])  # fmt:skip

    # Missing Freesurfer AND QSIPrep T1ws, or the user wants a DWI-based mask
    if not (has_qsiprep_t1w or has_freesurfer) or prefer_dwi_mask:
        desc += (
            "No T1w weighted images were available for masking, so a mask "
            "was estimated based on the b=0 images in the DWI data itself. "
        )
        extract_b0s = pe.Node(ExtractB0s(b0_threshold=b0_threshold), name="extract_b0s")
        mask_b0s = pe.Node(afni.Automask(outputtype="NIFTI_GZ"), name="mask_b0s")
        workflow.connect([
            (inputnode, extract_b0s, [
                ("dwi_file", "dwi_series"),
                ("bval_file", "bval_file"),
            ]),
            (extract_b0s, mask_b0s, [("b0_series", "in_file")]),
            (mask_b0s, outputnode, [("out_file", "dwi_mask")]),
            (inputnode, outputnode, [(field, field) for field in connect_from_inputnode]),
        ])  # fmt:skip

        workflow = clean_datasinks(workflow, qsirecon_suffix)
        return workflow, _get_status()

    # No data from QSIRecon was available, BUT we have freesurfer! register it and
    # get the brain, masks and possibly a to-MNI transform.
    # --> If has_freesurfer AND has qsiprep_t1w, the necessary files were created earlier
    elif has_freesurfer and not has_qsiprep_t1w:
        fs_source = pe.Node(
            nio.FreeSurferSource(subjects_dir=config.execution.fs_subjects_dir),
            name="fs_source",
        )
        # Register the FreeSurfer brain to the DWI reference
        desc += (
            "A brainmasked T1w image from FreeSurfer was registered to the "
            "preprocessed DWI data. Brainmasks from FreeSurfer were used in all "
            "subsequent reconstruction steps. "
        )

        # Move these fields to the buffernode
        _exchange_fields(
            FS_FILES_TO_REGISTER
            + [
                "acpc_brain_mask",
                "acpc_preproc",
                "fs_to_qsiprep_transform_mrtrix",
                "fs_to_qsiprep_transform_itk",
            ]
        )

        # Perform the registration and connect the outputs to buffernode
        # NOTE: using FreeSurfer "brain" image as acpc_preproc and aseg as the brainmask
        has_qsiprep_t1w = True
        register_fs_to_qsiprep_wf = init_register_fs_to_qsiprep_wf(
            use_qsiprep_reference_mask=False,
        )
        workflow.connect([
            (inputnode, fs_source, [("subject_id", "subject_id")]),
            (inputnode, register_fs_to_qsiprep_wf, [
                ("dwi_ref", "inputnode.qsiprep_reference_image"),
            ]),
            (fs_source, register_fs_to_qsiprep_wf, [
                (field, "inputnode." + field) for field in FS_FILES_TO_REGISTER
            ]),
            (register_fs_to_qsiprep_wf, buffernode, [
                ("outputnode.brain", "acpc_preproc"),
                ("outputnode.aseg", "acpc_brain_mask"),
                ("outputnode.fs_to_qsiprep_transform_mrtrix", "fs_to_qsiprep_transform_mrtrix"),
                ("outputnode.fs_to_qsiprep_transform_itk", "fs_to_qsiprep_transform_itk"),
            ] + [("outputnode." + field, field) for field in FS_FILES_TO_REGISTER],
            ),
        ])  # fmt:skip

    # Do we need to transform the 5tt hsvs from fsnative?
    if "mrtrix_5tt_hsvs" in extras_to_make and not has_qsiprep_5tt_hsvs:
        # Transform the 5tt image so it's registered to the QSIRecon AC-PC T1w
        config.loggers.workflow.info(
            "HSVS 5tt imaged will be registered to the " "QSIRecon dwiref image."
        )
        _exchange_fields(["qsiprep_5tt_hsvs"])
        if not has_freesurfer_5tt_hsvs:
            raise Exception("The 5tt image in fsnative should have been created by now")

        apply_header_to_5tt_hsvs = pe.Node(TransformHeader(), name="apply_header_to_5tt_hsvs")
        ds_qsiprep_5tt_hsvs = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                atlas="hsvs",
                suffix="dseg",
            ),
            name="ds_qsiprep_5tt_hsvs",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, apply_header_to_5tt_hsvs, [("fs_5tt_hsvs", "in_image")]),
            (apply_header_to_5tt_hsvs, buffernode, [("out_image", "qsiprep_5tt_hsvs")]),
            (apply_header_to_5tt_hsvs, ds_qsiprep_5tt_hsvs, [("out_image", "in_file")]),
        ])  # fmt:skip
        desc += "A hybrid surface/volume segmentation was created [Smith 2020]. "

    # If we have transforms to the template space, use them to get ROIs/atlases
    # if not has_qsiprep_t1w_transforms and has_qsiprep_t1w:
    #     desc += "In order to warp brain parcellations from template space into " \
    #         "alignment with the DWI data, the DWI-aligned FreeSurfer brain was " \
    #         "registered to template space. "

    #     # We now have qsiprep t1w and transforms!!
    #     has_qsiprep_t1w = has_qsiprep_t1w_transforms = True
    #     # Calculate the transforms here:
    #     has_qsiprep_t1w_transforms = True
    #     _exchange_fields(["acpc_to_template_xfm", "template_to_acpc_xfm"])
    #     t1_2_mni = pe.Node(
    #         get_t1w_registration_node(
    #             infant_mode, sloppy or not atlas_names, omp_nthreads),
    #         name="t1_2_mni")
    #     workflow.connect([
    #         (_get_source_node("acpc_preproc"), t1_2_mni, [("acpc_preproc", "moving_image")]),
    #         (t1_2_mni, buffernode, [
    #             ("composite_transform", "acpc_to_template_xfm"),
    #             ("inverse_composite_transform", "template_to_acpc_xfm")
    #         ])
    #     ])  # fmt:skip
    #     # TODO: add datasinks here

    # Check the status of the T1wACPC-to-template transforms
    if needs_t1w_transform:
        if has_qsiprep_t1w_transforms:
            config.loggers.workflow.info("Found T1w-to-template transforms from QSIRecon")
            desc += (
                "T1w-based spatial normalization calculated during "
                "preprocessing was used to map atlases from template space into "
                "alignment with DWIs. "
            )
        else:
            raise Exception(
                "Reconstruction workflow requires a T1wACPC-to-template transform. "
                "None were found."
            )

    # Simply resample the T1w mask into the DWI resolution. This was the default
    # up to version 0.14.3
    if has_qsiprep_t1w and not prefer_dwi_mask:
        desc += (
            f"Brainmasks from {skull_strip_method} were used in all subsequent reconstruction "
            "steps. "
        )
        # Resample anat mask
        resample_mask = pe.Node(
            ants.ApplyTransforms(
                dimension=3,
                transforms=["identity"],
                interpolation="NearestNeighbor",
            ),
            name="resample_mask",
        )

        workflow.connect([
            (inputnode, resample_mask, [
                ("acpc_brain_mask", "input_image"),
                ("dwi_ref", "reference_image"),
            ]),
            (resample_mask, buffernode, [("output_image", "dwi_mask")]),
        ])  # fmt:skip

    if has_qsiprep_t1w_transforms:
        config.loggers.workflow.info("Transforming ODF ROIs into DWI space for visual report.")
        # Resample ROI targets to DWI resolution for ODF plotting
        crossing_rois_file = pkgrf("qsirecon", "data/crossing_rois.nii.gz")
        odf_rois = pe.Node(
            ants.ApplyTransforms(interpolation="MultiLabel", dimension=3), name="odf_rois"
        )
        odf_rois.inputs.input_image = crossing_rois_file
        workflow.connect([
            (_get_source_node("template_to_acpc_xfm"), odf_rois, [
                ("template_to_acpc_xfm", "transforms"),
            ]),
            (inputnode, odf_rois, [("dwi_file", "reference_image")]),
            (odf_rois, buffernode, [("output_image", "odf_rois")]),
        ])  # fmt:skip

        # Similarly, if we need atlases, transform them into DWI space
        if atlas_configs:
            atlas_str = describe_atlases(sorted(list(atlas_configs.keys())))
            desc += (
                f"The following atlases were used in the workflow: {atlas_str}. "
                "Cortical parcellations were mapped from template space to DWIS "
                "using the T1w-based spatial normalization. "
            )

            # Resample all atlases to dwi_file's resolution
            prepare_atlases = pe.Node(
                WarpConnectivityAtlases(atlas_configs=atlas_configs),
                name="prepare_atlases",
                run_without_submitting=True,
            )
            workflow.connect([
                (inputnode, prepare_atlases, [("dwi_file", "reference_image")]),
                (prepare_atlases, buffernode, [("atlas_configs", "atlas_configs")]),
            ])  # fmt:skip

        for atlas in atlas_configs.keys():
            ds_atlas = pe.Node(
                DerivativesDataSink(
                    dismiss_entities=("desc",),
                    seg=atlas,
                    suffix="dseg",
                    compress=True,
                ),
                name=f"ds_atlas_{atlas}",
                run_without_submitting=True,
            )
            ds_atlas_mifs = pe.Node(
                DerivativesDataSink(
                    dismiss_entities=("desc",),
                    seg=atlas,
                    suffix="dseg",
                    extension=".mif.gz",
                    compress=True,
                ),
                name=f"ds_atlas_mifs_{atlas}",
                run_without_submitting=True,
            )
            ds_atlas_mrtrix_lut = pe.Node(
                DerivativesDataSink(
                    dismiss_entities=("desc",),
                    seg=atlas,
                    suffix="dseg",
                    extension=".txt",
                ),
                name=f"ds_atlas_mrtrix_lut_{atlas}",
                run_without_submitting=True,
            )
            ds_atlas_orig_lut = pe.Node(
                DerivativesDataSink(
                    dismiss_entities=("desc",),
                    seg=atlas,
                    suffix="dseg",
                    extension=".txt",
                ),
                name=f"ds_atlas_orig_lut_{atlas}",
                run_without_submitting=True,
            )
            workflow.connect([
                (prepare_atlases, ds_atlas, [(
                    ("atlas_configs", _get_resampled, atlas, "dwi_resolution_file"), "in_file"),
                ]),
                (prepare_atlases, ds_atlas_mifs, [(
                    ("atlas_configs", _get_resampled, atlas, "dwi_resolution_mif"), "in_file"),
                ]),
                (prepare_atlases, ds_atlas_mrtrix_lut, [(
                    ("atlas_configs", _get_resampled, atlas, "mrtrix_lut"), "in_file"),
                ]),
                (prepare_atlases, ds_atlas_orig_lut, [(
                    ("atlas_configs", _get_resampled, atlas, "orig_lut"), "in_file"),
                ]),
            ])  # fmt:skip

        # Fill in the atlas datasinks
        for node in workflow.list_node_names():
            node_suffix = node.split(".")[-1]
            if node_suffix.startswith("ds_atlas_"):
                workflow.connect([
                    (inputnode, workflow.get_node(node), [("dwi_file", "source_file")]),
                ])  # fmt:skip

    if "mrtrix_5tt_hsvs" in extras_to_make and not has_qsiprep_5tt_hsvs:
        raise Exception("Unable to create a 5tt HSV image given input data.")

    workflow.__desc__ = desc + "\n"

    # Directly connect anything from the inputs that we haven't created here
    workflow.connect([
        (inputnode, outputnode, [(name, name) for name in connect_from_inputnode]),
        (buffernode, outputnode, [(name, name) for name in connect_from_buffernode])
    ])  # fmt:skip

    workflow = clean_datasinks(workflow, qsirecon_suffix)
    return workflow, _get_status()


def _get_first(item):
    if isinstance(item, (list, tuple)):
        return item[0]
    return item


def _get_resampled(atlas_configs, atlas_name, to_retrieve):
    return atlas_configs[atlas_name][to_retrieve]


def init_output_grid_wf() -> Workflow:
    """Generate a non-oblique, uniform voxel-size grid around a brain."""
    workflow = Workflow(name="output_grid_wf")
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["template_image", "input_image"]), name="inputnode"
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["grid_image"]), name="outputnode")
    # Create the output reference grid_image
    if config.workflow.output_resolution is None:
        voxel_size = traits.Undefined
    else:
        voxel_size = config.workflow.output_resolution
    padding = 4 if config.workflow.infant else 8

    autobox_template = pe.Node(
        afni.Autobox(outputtype="NIFTI_GZ", padding=padding), name="autobox_template"
    )
    deoblique_autobox = pe.Node(
        afni.Warp(outputtype="NIFTI_GZ", deoblique=True), name="deoblique_autobox"
    )
    voxel_size_chooser = pe.Node(
        VoxelSizeChooser(voxel_size=voxel_size), name="voxel_size_chooser"
    )
    resample_to_voxel_size = pe.Node(
        afni.Resample(outputtype="NIFTI_GZ"), name="resample_to_voxel_size"
    )

    workflow.connect([
        (inputnode, autobox_template, [("template_image", "in_file")]),
        (autobox_template, deoblique_autobox, [("out_file", "in_file")]),
        (deoblique_autobox, resample_to_voxel_size, [("out_file", "in_file")]),
        (resample_to_voxel_size, outputnode, [("out_file", "grid_image")]),
        (inputnode, voxel_size_chooser, [("input_image", "input_image")]),
        (voxel_size_chooser, resample_to_voxel_size, [(("voxel_size", _tupleize), "voxel_size")])
    ])  # fmt:skip

    return workflow


def _tupleize(value):
    # Nipype did not like having a Tuple output trait
    return (value, value, value)
