# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from pathlib import Path

import nipype.interfaces.io as nio
from nipype.interfaces import afni, ants, mrtrix3
from nipype.interfaces import utility as niu
from nipype.interfaces.base import traits
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...data import load as load_data
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
from ...interfaces.utils import (
    AtlasLUTs,
    ConformAtlas,
    RecombineAtlasConfigs,
)
from ...utils.bids import clean_datasinks
from ...utils.boilerplate import describe_atlases

# Required freesurfer files for mrtrix's HSV 5tt generation
HSV_REQUIREMENTS = [
    'mri/aparc+aseg.mgz',
    'mri/brainmask.mgz',
    'mri/norm.mgz',
    'mri/transforms/talairach.xfm',
    'surf/lh.white',
    'surf/lh.pial',
    'surf/rh.white',
    'surf/rh.pial',
]


def init_highres_recon_anatomical_wf(
    subject_id,
    session_id,
    extras_to_make,
    status,
    name='recon_anatomical_wf',
):
    """Pass through preprocessed anatomical data and build FreeSurfer-derived extras.

    This is the subject-level (or session-level) anatomical workflow.
    It is built once per anatomical image and shared by every DWI paired with that image.
    Files that depend on a DWI's grid are handled downstream by
    :func:`init_dwi_recon_anatomical_workflow`.

    The workflow does the following:

    1. If neither a preprocessed anatomical image nor a FreeSurfer directory is available,
       an empty workflow is returned.
       An error is raised instead if the recon spec requires a HSVS 5tt image.
    2. Otherwise, every field collected by
       :func:`~qsirecon.utils.bids.collect_anatomical_data` is forwarded from the inputnode
       to the outputnode unchanged.
    3. If ``'mrtrix_5tt_hsvs'`` is in ``extras_to_make``, a hybrid surface/volume
       segmentation (HSVS) 5tt image is created from the FreeSurfer directory in
       FreeSurfer's native space.
       If a preprocessed anatomical image is also available, FreeSurfer's ``brain`` and
       ``aseg`` are rigidly registered to it (see :func:`init_register_fs_to_qsiprep_wf`)
       and the 5tt image is moved into AC-PC space with the same transform.

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label, without the ``sub-`` prefix.
    session_id : None or :obj:`bids.layout.Query` or :obj:`list`
        Session filter used to locate the FreeSurfer directory.
        See :func:`~qsirecon.utils.bids.collect_anatomical_data` for the accepted values.
    extras_to_make : :obj:`list` of :obj:`str`
        The ``anatomical`` entries of the recon spec.
        Currently only ``'mrtrix_5tt_hsvs'`` is recognized.
    status : :obj:`dict`
        The status dictionary returned by
        :func:`~qsirecon.utils.bids.collect_anatomical_data`.
        It must contain ``has_qsiprep_anat`` and ``has_qsiprep_anat_transforms``.
    name : :obj:`str`, optional
        Name of the workflow.

    Inputs
    ------
    acpc_preproc, acpc_brain_mask, acpc_seg, acpc_aseg, acpc_aparc, acpc_csf_probseg, \
    acpc_gm_probseg, acpc_wm_probseg, orig_to_acpc_xfm, acpc_to_template_xfm, \
    template_to_acpc_xfm
        The files collected by :func:`~qsirecon.utils.bids.collect_anatomical_data`,
        each of which may be ``None``.
        ``acpc_preproc`` is the preprocessed anatomical image, which may be a T1w or a T2w.

    Outputs
    -------
    acpc_preproc, acpc_brain_mask, acpc_seg, acpc_aseg, acpc_aparc, acpc_csf_probseg, \
    acpc_gm_probseg, acpc_wm_probseg, orig_to_acpc_xfm, acpc_to_template_xfm, \
    template_to_acpc_xfm
        The inputs, forwarded unchanged.
    brain, aseg
        FreeSurfer's ``brain.mgz`` and ``aseg.mgz``.
        When a HSVS 5tt image was requested and a preprocessed anatomical image is
        available, their headers are adjusted so that they are aligned with
        ``acpc_preproc``.
        When a FreeSurfer directory is available but no preprocessed anatomical image is,
        they are forwarded in FreeSurfer's native space for
        :func:`init_dwi_recon_anatomical_workflow` to register to each DWI.
        Otherwise they are not set.
    fs_to_qsiprep_transform_itk, fs_to_qsiprep_transform_mrtrix
        The rigid transform from FreeSurfer native space to AC-PC space, in ITK and
        MRtrix3 formats.
        Only set when a HSVS 5tt image was requested and a preprocessed anatomical image
        is available.
    fs_5tt_hsvs
        The HSVS 5tt image in FreeSurfer native space, if requested.
    qsiprep_5tt_hsvs
        The HSVS 5tt image aligned to ``acpc_preproc``, if requested and a preprocessed
        anatomical image is available.
        Without a preprocessed anatomical image the 5tt image is instead aligned to each
        DWI by :func:`init_dwi_recon_anatomical_workflow`, and no anatomical-named 5tt
        derivatives are written here.

    Returns
    -------
    workflow : :obj:`niworkflows.engine.workflows.LiterateWorkflow`
        The workflow.
    status : :obj:`dict`
        ``status`` with ``has_freesurfer``, ``has_freesurfer_5tt_hsvs``, and
        ``has_qsiprep_5tt_hsvs`` added.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=anatomical_workflow_outputs),
        name='outputnode',
    )
    workflow.__desc__ = ''
    qsirecon_suffix = ''

    # If there is no high-res anat data in the inputs there may still be an image available
    # from freesurfer. Check for it:
    freesurfer_dir = config.execution.fs_subjects_dir
    subject_freesurfer_path = find_fs_path(freesurfer_dir, subject_id, session_id)
    status['has_freesurfer'] = subject_freesurfer_path is not None
    status['has_qsiprep_5tt_hsvs'] = False
    status['has_freesurfer_5tt_hsvs'] = False

    # If no high-res are available, we're done here
    if not status['has_qsiprep_anat'] and not status['has_freesurfer']:
        config.loggers.workflow.warning(
            f'No high-res anatomical data available directly in recon inputs for {subject_id}.'
        )
        # If a 5tt image is needed, this is an error
        if 'mrtrix_5tt_hsvs' in extras_to_make:
            raise Exception('FreeSurfer data is required to make HSVS 5tt image.')

        workflow.add_nodes([outputnode])
        workflow = clean_datasinks(workflow, qsirecon_suffix)
        return workflow, status

    config.loggers.workflow.info(
        f'Found high-res anatomical data in preprocessed inputs for {subject_id}.'
    )
    workflow.connect([
        (inputnode, outputnode, [
            (name, name) for name in qsiprep_highres_anatomical_ingressed_fields
        ]),
    ])  # fmt:skip

    # grab un-coregistered freesurfer images for later use
    if status['has_freesurfer']:
        config.loggers.workflow.info(
            f'Freesurfer directory {subject_freesurfer_path} exists for {subject_id}'
        )
        fs_source = pe.Node(
            nio.FreeSurferSource(
                subjects_dir=str(subject_freesurfer_path.parent),
                subject_id=subject_freesurfer_path.name,
            ),
            name='fs_source',
        )
        if not status['has_qsiprep_anat']:
            # There is no preprocessed anatomical image to register FreeSurfer to here.
            # Forward the un-registered files so that init_dwi_recon_anatomical_workflow can
            # register them to each DWI's reference image.
            workflow.connect([
                (fs_source, outputnode, [(field, field) for field in FS_FILES_TO_REGISTER]),
            ])  # fmt:skip

    # Do we need to calculate anything else on the fly
    if 'mrtrix_5tt_hsvs' in extras_to_make:
        # Check for specific files needed for Hybrid Surface and Volume Segmentation (HSVS).
        missing_fs_hsvs_files = check_hsv_inputs(Path(subject_freesurfer_path))
        if missing_fs_hsvs_files:
            raise Exception(' '.join(missing_fs_hsvs_files) + 'are missing: unable to make a HSV.')

        config.loggers.workflow.info('FreeSurfer data will be used to create a HSVS 5tt image.')
        status['has_freesurfer_5tt_hsvs'] = True
        create_5tt_hsvs = pe.Node(
            GenerateMasked5tt(
                algorithm='hsvs',
                in_file=str(subject_freesurfer_path),
                nthreads=config.nipype.omp_nthreads,
            ),
            name='create_5tt_hsvs',
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([(create_5tt_hsvs, outputnode, [('out_file', 'fs_5tt_hsvs')])])

        # Transform the 5tt image so it's registered to the preprocessed AC-PC anatomical
        if not status['has_qsiprep_anat']:
            # The derivatives are named after an anatomical image, so without one the 5tt
            # image is only written once it has been aligned to a DWI, by
            # init_dwi_recon_anatomical_workflow.
            config.loggers.workflow.info(
                'No preprocessed anatomical image is available, so the HSVS 5tt image will be '
                'registered to each DWI instead.'
            )
        else:
            config.loggers.workflow.info(
                'HSVS 5tt image will be registered to the preprocessed anatomical image.'
            )
            status['has_qsiprep_5tt_hsvs'] = True
            ds_qsiprep_5tt_hsvs = pe.Node(
                DerivativesDataSink(
                    dismiss_entities=('desc',),
                    space='ACPC',
                    seg='hsvs',
                    suffix='probseg',
                    extension='.nii.gz',
                    compress=True,
                ),
                name='ds_qsiprep_5tt_hsvs',
                run_without_submitting=True,
            )
            ds_fs_5tt_hsvs = pe.Node(
                DerivativesDataSink(
                    dismiss_entities=('desc',),
                    seg='hsvs',
                    space='fsnative',
                    suffix='probseg',
                    extension='.mif.gz',
                    compress=True,
                ),
                name='ds_fs_5tt_hsvs',
                run_without_submitting=True,
            )
            register_fs_to_qsiprep_wf = init_register_fs_to_qsiprep_wf(
                use_qsiprep_reference_mask=True,
            )
            apply_header_to_5tt = pe.Node(TransformHeader(), name='apply_header_to_5tt')
            workflow.connect([
                (inputnode, ds_fs_5tt_hsvs, [('acpc_preproc', 'source_file')]),
                (inputnode, ds_qsiprep_5tt_hsvs, [('acpc_preproc', 'source_file')]),
                (create_5tt_hsvs, ds_fs_5tt_hsvs, [('out_file', 'in_file')]),
                (inputnode, register_fs_to_qsiprep_wf, [
                    ('acpc_preproc', 'inputnode.qsiprep_reference_image'),
                    ('acpc_brain_mask', 'inputnode.qsiprep_reference_mask'),
                ]),
                (fs_source, register_fs_to_qsiprep_wf, [
                    (field, f'inputnode.{field}') for field in FS_FILES_TO_REGISTER
                ]),
                (register_fs_to_qsiprep_wf, outputnode, [
                    ('outputnode.fs_to_qsiprep_transform_mrtrix',
                        'fs_to_qsiprep_transform_mrtrix'),
                    ('outputnode.fs_to_qsiprep_transform_itk', 'fs_to_qsiprep_transform_itk')
                ] + [
                    (f'outputnode.{field}', field) for field in FS_FILES_TO_REGISTER
                ]),
                (create_5tt_hsvs, apply_header_to_5tt, [('out_file', 'in_image')]),
                (register_fs_to_qsiprep_wf, apply_header_to_5tt, [
                    ('outputnode.fs_to_qsiprep_transform_mrtrix', 'transform_file'),
                ]),
                (apply_header_to_5tt, outputnode, [('out_image', 'qsiprep_5tt_hsvs')]),
                (apply_header_to_5tt, ds_qsiprep_5tt_hsvs, [('out_image', 'in_file')]),
            ])  # fmt:skip

            workflow.__desc__ += 'A hybrid surface/volume segmentation was created [Smith 2020].'

    workflow = clean_datasinks(workflow, qsirecon_suffix)
    return workflow, status


def check_hsv_inputs(subj_fs_path):
    """Determine if a FreeSurfer directory has the required files for HSV."""
    return [
        requirement
        for requirement in HSV_REQUIREMENTS
        if not (subj_fs_path / requirement).exists()
    ]


def _check_zipped_unzipped(path_to_check):
    """Check to see if a path exists and warn if it's gzipped."""

    exists = False
    if path_to_check.exists():
        exists = True
    if path_to_check.name.endswith('.gz'):
        nonzipped = str(path_to_check)[:-3]
        if Path(nonzipped).exists():
            config.loggers.workflow.warn(
                'A Non-gzipped input nifti file was found. Consider gzipping %s', nonzipped
            )
            exists = True
    config.loggers.workflow.info(f'CHECKING {path_to_check}: {exists}')
    return exists


def init_register_fs_to_qsiprep_wf(
    use_qsiprep_reference_mask=False,
    name='register_fs_to_qsiprep_wf',
):
    """Rigidly register FreeSurfer's brain to a reference image and realign FreeSurfer files.

    FreeSurfer's ``brain.mgz`` is converted to NIfTI and registered to
    ``qsiprep_reference_image`` with a rigid ANTs registration driven by the Mattes
    mutual-information metric, so the reference may be a T1w, a T2w, or a DWI reference
    image.
    The transform is converted to MRtrix3 format and applied to the headers of the
    FreeSurfer images listed in
    :data:`~qsirecon.interfaces.interchange.FS_FILES_TO_REGISTER` (``brain`` and ``aseg``),
    so the images are aligned in world coordinates without interpolation.

    Parameters
    ----------
    use_qsiprep_reference_mask : :obj:`bool`, optional
        Restrict the registration metric to voxels inside ``qsiprep_reference_mask``.
    name : :obj:`str`, optional
        Name of the workflow.

    Inputs
    ------
    brain, aseg
        FreeSurfer's ``brain.mgz`` and ``aseg.mgz``.
    qsiprep_reference_image
        The fixed image: the preprocessed anatomical image, or the DWI reference image.
    qsiprep_reference_mask
        Brain mask of the fixed image.
        Only used when ``use_qsiprep_reference_mask`` is ``True``.

    Outputs
    -------
    brain, aseg
        The input images with headers adjusted into alignment with the reference image.
    fs_to_qsiprep_transform_itk
        Rigid transform from FreeSurfer native space to the reference image, in ITK format.
    fs_to_qsiprep_transform_mrtrix
        The same transform in MRtrix3 format.
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=FS_FILES_TO_REGISTER + ['qsiprep_reference_image', 'qsiprep_reference_mask']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=FS_FILES_TO_REGISTER
            + ['fs_to_qsiprep_transform_itk', 'fs_to_qsiprep_transform_mrtrix']
        ),
        name='outputnode',
    )
    workflow = Workflow(name=name)
    workflow.__desc__ = 'FreeSurfer outputs were registered to the QSIRecon outputs.'

    # Convert the freesurfer inputs so we can register them with ANTs
    convert_fs_brain = pe.Node(
        mrtrix3.MRConvert(out_file='fs_brain.nii', args='-strides -1,-2,3'),
        name='convert_fs_brain',
    )

    # Register the brain to the QSIRecon reference
    ants_settings = load_data('freesurfer_to_qsiprep.json')
    register_to_qsiprep = pe.Node(
        ants.Registration(from_file=ants_settings), name='register_to_qsiprep'
    )

    # If there is a mask for the QSIRecon reference image, use it
    if use_qsiprep_reference_mask:
        workflow.connect(inputnode, 'qsiprep_reference_mask',
                         register_to_qsiprep, 'fixed_image_masks')  # fmt:skip

    # The more recent ANTs mat format isn't compatible with transformconvert.
    # So convert it to ANTs text format with ConvertTransform
    convert_ants_transform = pe.Node(
        ConvertTransformFile(dimension=3), name='convert_ants_transform'
    )

    # Convert from ANTs text format to MRtrix3 format
    convert_ants_to_mrtrix_transform = pe.Node(
        ITKTransformConvert(), name='convert_ants_to_mrtrix_transform'
    )

    # Adjust the headers of all the input images so they're aligned to the qsiprep ref
    transform_nodes = {}
    for image_name in FS_FILES_TO_REGISTER:
        transform_nodes[image_name] = pe.Node(TransformHeader(), name='transform_' + image_name)
        workflow.connect([
            (inputnode, transform_nodes[image_name], [(image_name, 'in_image')]),
            (convert_ants_to_mrtrix_transform,
             transform_nodes[image_name], [('out_transform', 'transform_file')]),
            (transform_nodes[image_name], outputnode, [('out_image', image_name)])
        ])  # fmt:skip

    workflow.connect([
        (inputnode, convert_fs_brain, [('brain', 'in_file')]),
        (inputnode, register_to_qsiprep, [('qsiprep_reference_image', 'fixed_image')]),
        (convert_fs_brain, register_to_qsiprep, [('out_file', 'moving_image')]),
        (register_to_qsiprep, convert_ants_transform, [
            (('forward_transforms', _get_first), 'in_transform'),
        ]),
        (register_to_qsiprep, outputnode, [
            ('composite_transform', 'fs_to_qsiprep_transform_itk'),
        ]),
        (convert_ants_transform, convert_ants_to_mrtrix_transform, [
            ('out_transform', 'in_transform'),
        ]),
        (convert_ants_to_mrtrix_transform, outputnode, [
            ('out_transform', 'fs_to_qsiprep_transform_mrtrix'),
        ]),
    ])  # fmt:skip

    return workflow


def init_dwi_recon_anatomical_workflow(
    atlas_configs,
    has_qsiprep_5tt_hsvs,
    needs_anat_transform,
    has_freesurfer_5tt_hsvs,
    has_qsiprep_anat,
    has_qsiprep_anat_transforms,
    has_freesurfer,
    extras_to_make,
    name,
    template_output_space,
    prefer_dwi_mask=False,
):
    """Build the DWI-specific anatomical data for one preprocessed DWI series.

    The subject-level anatomical workflow (:func:`init_highres_recon_anatomical_wf`)
    provides files in AC-PC space.
    This workflow derives the files that depend on the DWI's own grid, and it substitutes
    FreeSurfer or DWI-derived files when no preprocessed anatomical image is available.

    The workflow does the following:

    1. If a template space is known (``template_output_space``), builds
       ``resampling_template``: a non-oblique grid at the requested output resolution that
       covers the brain in template space (see :func:`init_output_grid_wf`).
       Without a template space, ``resampling_template`` is left undefined.
    2. Chooses a brain mask (``dwi_mask``):

       - If neither a preprocessed anatomical image nor a FreeSurfer directory is
         available, or ``prefer_dwi_mask`` is ``True``, the mask is estimated from the
         b=0 volumes of the DWI with AFNI's ``3dAutomask``.
         Nothing else is built in this case.
       - If only FreeSurfer is available, the native-space ``brain`` and ``aseg`` from the
         inputnode are rigidly registered to the DWI reference image
         (see :func:`init_register_fs_to_qsiprep_wf`) and take the place of
         ``acpc_preproc`` and ``acpc_brain_mask``.
       - Otherwise, the preprocessed brain mask is resampled onto the DWI grid.

    3. If a HSVS 5tt image exists in FreeSurfer native space but was not aligned to a
       preprocessed anatomical image upstream (the FreeSurfer-only case), its header is
       adjusted with the FreeSurfer-to-DWI transform from step 2 and the result is written
       as a derivative of the DWI.
    4. If transforms between AC-PC and template space are available, the ROIs used for
       ODF plots in the reports and any requested atlases are warped from template space
       onto the DWI grid (see :func:`init_warp_atlases_wf`).

    Parameters
    ----------
    atlas_configs : :obj:`dict`
        Atlas configurations from :func:`~qsirecon.utils.atlases.collect_atlases`,
        keyed by atlas name.
        May be empty.
    has_qsiprep_5tt_hsvs : :obj:`bool`
        Whether a HSVS 5tt image aligned to the preprocessed anatomical image exists.
    needs_anat_transform : :obj:`bool`
        Whether the recon workflow requires transforms between AC-PC and template space.
        An error is raised if they are required but unavailable.
    has_freesurfer_5tt_hsvs : :obj:`bool`
        Whether a HSVS 5tt image in FreeSurfer native space exists.
    has_qsiprep_anat : :obj:`bool`
        Whether a preprocessed anatomical image (T1w or T2w) and its brain mask are
        available.
    has_qsiprep_anat_transforms : :obj:`bool`
        Whether transforms between AC-PC and template space are available.
    has_freesurfer : :obj:`bool`
        Whether a FreeSurfer directory is available for the subject.
    extras_to_make : :obj:`list` of :obj:`str`
        The ``anatomical`` entries of the recon spec.
    name : :obj:`str`
        Name of the workflow.
    template_output_space : :obj:`str` or None
        Name of the template that the transforms map to
        (for example ``'MNI152NLin2009cAsym'``), or ``None`` if there are no transforms.
    prefer_dwi_mask : :obj:`bool`, optional
        Estimate the brain mask from the DWI even when an anatomical mask is available.

    Inputs
    ------
    dwi_file, bval_file, dwi_ref
        The preprocessed DWI series, its b-values, and its reference image.
        These, and the other fields of
        :class:`~qsirecon.interfaces.ingress.QSIPrepDWIIngress`, are forwarded to the
        outputnode.
    acpc_preproc, acpc_brain_mask, template_to_acpc_xfm, fs_5tt_hsvs, brain, aseg
        Outputs of :func:`init_highres_recon_anatomical_wf` that are used here.
        ``brain`` and ``aseg`` are only read in the FreeSurfer-only case, where they are
        in FreeSurfer's native space.
        The remaining outputs of that workflow are forwarded unchanged.

    Outputs
    -------
    dwi_mask
        Brain mask on the DWI grid.
    resampling_template
        Reference grid for outputs in template space.
    odf_rois
        Crossing-fiber ROIs for ODF plots, on the DWI grid.
        Only set when transforms are available.
    atlas_configs
        ``atlas_configs`` with each atlas resampled onto the DWI grid.
        Only set when transforms and atlases are available.
    acpc_preproc, acpc_brain_mask, brain, aseg, fs_to_qsiprep_transform_itk, \
    fs_to_qsiprep_transform_mrtrix
        Replaced with FreeSurfer-derived files when only FreeSurfer is available.
        Otherwise forwarded from the inputs.
    qsiprep_5tt_hsvs
        The HSVS 5tt image aligned to the DWI, when it was not already aligned upstream.
        Otherwise forwarded from the inputs.

    Returns
    -------
    workflow : :obj:`niworkflows.engine.workflows.LiterateWorkflow`
        The workflow.
    status : :obj:`dict`
        The availability flags after this workflow has run, with the keys
        ``has_qsiprep_5tt_hsvs``, ``has_freesurfer_5tt_hsvs``, ``has_qsiprep_anat``,
        ``has_qsiprep_anat_transforms``, and ``template_output_space``.
        ``has_qsiprep_anat`` becomes ``True`` when FreeSurfer files were substituted for
        the preprocessed anatomical image.
        These keys are merged into the ``inputs_dict`` that reconstruction nodes receive.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = '\n\n#### Anatomical data for DWI reconstruction\n\n'

    # Inputnode holds data from the subject-level anatomical workflow
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='inputnode',
    )
    connect_from_inputnode = set(recon_workflow_input_fields)
    # Buffer to hold the anatomical files that are calculated here
    buffernode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='buffernode',
    )
    connect_from_buffernode = set()
    b0_threshold = config.workflow.b0_threshold
    qsirecon_suffix = ''

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
    _exchange_fields(['dwi_mask', 'atlas_configs', 'odf_rois', 'resampling_template'])

    outputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='outputnode',
    )

    skull_strip_method = 'antsBrainExtraction'

    def _get_status():
        return {
            'has_qsiprep_5tt_hsvs': has_qsiprep_5tt_hsvs,
            'has_freesurfer_5tt_hsvs': has_freesurfer_5tt_hsvs,
            'has_qsiprep_anat': has_qsiprep_anat,
            'has_qsiprep_anat_transforms': has_qsiprep_anat_transforms,
            'template_output_space': template_output_space,
        }

    # The template-space resampling grid needs a known template, which comes from the
    # ACPC-to-template transforms. Without them (e.g., --anat-modality none), nothing can be
    # mapped to template space, so resampling_template is left undefined.
    if template_output_space is None:
        config.loggers.workflow.info(
            'No template space is known for this DWI, so no template-space resampling grid '
            'will be created.'
        )
    else:
        get_template = pe.Node(
            GetTemplate(
                template_name=template_output_space,
            ),
            name='get_template',
        )
        mask_template = pe.Node(
            afni.Calc(expr='a*b', outputtype='NIFTI_GZ'),
            name='mask_template',
        )
        reorient_to_lps = pe.Node(
            afni.Resample(orientation='RAI', outputtype='NIFTI_GZ'),
            name='reorient_to_lps',
        )

        reference_grid_wf = init_output_grid_wf()
        workflow.connect([
            (get_template, mask_template, [
                ('template_file', 'in_file_a'),
                ('mask_file', 'in_file_b'),
            ]),
            (mask_template, reorient_to_lps, [('out_file', 'in_file')]),
            (inputnode, reference_grid_wf, [('dwi_ref', 'inputnode.input_image')]),
            (reorient_to_lps, reference_grid_wf, [('out_file', 'inputnode.template_image')]),
            (reference_grid_wf, buffernode, [('outputnode.grid_image', 'resampling_template')]),
        ])  # fmt:skip

    # Set when FreeSurfer is registered to the DWI below (the FreeSurfer-only case).
    register_fs_to_qsiprep_wf = None

    # Missing FreeSurfer AND a preprocessed anatomical, or the user wants a DWI-based mask
    if not (has_qsiprep_anat or has_freesurfer) or prefer_dwi_mask:
        workflow.__desc__ += (
            'No anatomical images were available for masking, so a mask '
            'was estimated based on the b=0 images in the DWI data itself. '
        )
        extract_b0s = pe.Node(ExtractB0s(b0_threshold=b0_threshold), name='extract_b0s')
        mask_b0s = pe.Node(afni.Automask(outputtype='NIFTI_GZ'), name='mask_b0s')
        workflow.connect([
            (inputnode, extract_b0s, [
                ('dwi_file', 'dwi_series'),
                ('bval_file', 'bval_file'),
            ]),
            (extract_b0s, mask_b0s, [('b0_series', 'in_file')]),
            (inputnode, outputnode, [(field, field) for field in connect_from_inputnode]),
        ])  # fmt:skip

        ds_dwi_mask = pe.Node(
            DerivativesDataSink(
                dismiss_entities=('desc',),
                datatype='dwi',
                desc='brain',
                suffix='mask',
                compress=True,
            ),
            name='ds_dwi_mask',
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, ds_dwi_mask, [('dwi_file', 'source_file')]),
            (mask_b0s, ds_dwi_mask, [('out_file', 'in_file')]),
            (ds_dwi_mask, outputnode, [('out_file', 'dwi_mask')]),
        ])  # fmt:skip

        workflow = clean_datasinks(workflow, qsirecon_suffix)
        return workflow, _get_status()

    # No data from QSIRecon was available, BUT we have freesurfer! register it and
    # get the brain, masks and possibly a to-MNI transform.
    # --> If has_freesurfer AND has_qsiprep_anat, the necessary files were created earlier
    elif has_freesurfer and not has_qsiprep_anat:
        # Register the FreeSurfer brain to the DWI reference. The un-registered FreeSurfer
        # files arrive on the inputnode from init_highres_recon_anatomical_wf.
        workflow.__desc__ += (
            'A brainmasked T1w image from FreeSurfer was registered to the '
            'preprocessed DWI data. Brainmasks from FreeSurfer were used in all '
            'subsequent reconstruction steps. '
        )

        # Move these fields to the buffernode
        _exchange_fields(
            FS_FILES_TO_REGISTER
            + [
                'acpc_brain_mask',
                'acpc_preproc',
                'fs_to_qsiprep_transform_mrtrix',
                'fs_to_qsiprep_transform_itk',
            ]
        )

        # Perform the registration and connect the outputs to buffernode
        # NOTE: using FreeSurfer "brain" image as acpc_preproc and aseg as the brainmask
        has_qsiprep_anat = True
        register_fs_to_qsiprep_wf = init_register_fs_to_qsiprep_wf(
            use_qsiprep_reference_mask=False,
        )
        workflow.connect([
            (inputnode, register_fs_to_qsiprep_wf, [
                ('dwi_ref', 'inputnode.qsiprep_reference_image'),
            ] + [
                (field, 'inputnode.' + field) for field in FS_FILES_TO_REGISTER
            ]),
            (register_fs_to_qsiprep_wf, buffernode, [
                ('outputnode.brain', 'acpc_preproc'),
                ('outputnode.aseg', 'acpc_brain_mask'),
                ('outputnode.fs_to_qsiprep_transform_mrtrix', 'fs_to_qsiprep_transform_mrtrix'),
                ('outputnode.fs_to_qsiprep_transform_itk', 'fs_to_qsiprep_transform_itk'),
            ] + [('outputnode.' + field, field) for field in FS_FILES_TO_REGISTER],
            ),
        ])  # fmt:skip

    # Do we need to transform the 5tt hsvs from fsnative?
    if 'mrtrix_5tt_hsvs' in extras_to_make and not has_qsiprep_5tt_hsvs:
        # Transform the 5tt image so it's registered to the DWI reference image
        config.loggers.workflow.info(
            'HSVS 5tt image will be registered to the QSIRecon dwiref image.'
        )
        _exchange_fields(['qsiprep_5tt_hsvs'])
        if not has_freesurfer_5tt_hsvs:
            raise Exception('The 5tt image in fsnative should have been created by now')
        if register_fs_to_qsiprep_wf is None:
            raise Exception(
                'The HSVS 5tt image can only be aligned to the DWI after FreeSurfer has been '
                'registered to it, which requires FreeSurfer data and no preprocessed '
                'anatomical image.'
            )

        apply_header_to_5tt_hsvs = pe.Node(TransformHeader(), name='apply_header_to_5tt_hsvs')
        # The 5tt image is aligned to this DWI's reference image, so it is a DWI-specific
        # derivative and is named after the DWI.
        ds_qsiprep_5tt_hsvs = pe.Node(
            DerivativesDataSink(
                dismiss_entities=('desc',),
                seg='hsvs',
                suffix='probseg',
                compress=True,
            ),
            name='ds_qsiprep_5tt_hsvs',
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, apply_header_to_5tt_hsvs, [('fs_5tt_hsvs', 'in_image')]),
            # Take the transform from the registration workflow directly rather than from the
            # buffernode, which this node also feeds; going through it would make a cycle.
            (register_fs_to_qsiprep_wf, apply_header_to_5tt_hsvs, [
                ('outputnode.fs_to_qsiprep_transform_mrtrix', 'transform_file'),
            ]),
            (inputnode, ds_qsiprep_5tt_hsvs, [('dwi_file', 'source_file')]),
            (apply_header_to_5tt_hsvs, buffernode, [('out_image', 'qsiprep_5tt_hsvs')]),
            (apply_header_to_5tt_hsvs, ds_qsiprep_5tt_hsvs, [('out_image', 'in_file')]),
        ])  # fmt:skip
        # The aligned 5tt image now exists for this DWI, so downstream ACT workflows can use it.
        has_qsiprep_5tt_hsvs = True
        workflow.__desc__ += 'A hybrid surface/volume segmentation was created [Smith 2020]. '

    # Check the status of the ACPC-to-template transforms
    if needs_anat_transform:
        if has_qsiprep_anat_transforms:
            config.loggers.workflow.info('Found anatomical-to-template transforms from QSIPrep')
            workflow.__desc__ += (
                'The spatial normalization of the anatomical image calculated during '
                'preprocessing was used to map atlases from template space into '
                'alignment with DWIs. '
            )
        else:
            raise Exception(
                'Reconstruction workflow requires an ACPC-to-template transform. None were found.'
            )

    # Simply resample the anatomical brain mask into the DWI resolution. This was the default
    # up to version 0.14.3
    if has_qsiprep_anat and not prefer_dwi_mask:
        workflow.__desc__ += (
            f'Brain masks from {skull_strip_method} were used in all subsequent reconstruction '
            'steps. '
        )
        # Resample anat mask
        resample_mask = pe.Node(
            ants.ApplyTransforms(
                dimension=3,
                transforms=['identity'],
                interpolation='NearestNeighbor',
            ),
            name='resample_mask',
        )
        if register_fs_to_qsiprep_wf is not None:
            # FreeSurfer-only: the mask is the registered aseg. Take it from the registration
            # workflow rather than the buffernode, which this branch also feeds (cycle).
            mask_source = (register_fs_to_qsiprep_wf, 'outputnode.aseg')
        else:
            mask_source = (inputnode, 'acpc_brain_mask')

        workflow.connect([
            (mask_source[0], resample_mask, [(mask_source[1], 'input_image')]),
            (inputnode, resample_mask, [('dwi_ref', 'reference_image')]),
        ])  # fmt:skip

        ds_resampled_mask = pe.Node(
            DerivativesDataSink(
                dismiss_entities=('desc',),
                datatype='dwi',
                desc='brain',
                suffix='mask',
                compress=True,
            ),
            name='ds_resampled_mask',
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, ds_resampled_mask, [('dwi_file', 'source_file')]),
            (resample_mask, ds_resampled_mask, [('output_image', 'in_file')]),
            (ds_resampled_mask, buffernode, [('out_file', 'dwi_mask')]),
        ])  # fmt:skip

    if has_qsiprep_anat_transforms:
        config.loggers.workflow.info('Transforming ODF ROIs into DWI space for visual report.')
        # Resample ROI targets to DWI resolution for ODF plotting
        crossing_rois_file = load_data('crossing_rois.nii.gz')
        odf_rois = pe.Node(
            ants.ApplyTransforms(interpolation='MultiLabel', dimension=3), name='odf_rois'
        )
        odf_rois.inputs.input_image = crossing_rois_file
        workflow.connect([
            (_get_source_node('template_to_acpc_xfm'), odf_rois, [
                ('template_to_acpc_xfm', 'transforms'),
            ]),
            (inputnode, odf_rois, [('dwi_file', 'reference_image')]),
            (odf_rois, buffernode, [('output_image', 'odf_rois')]),
        ])  # fmt:skip

        # Similarly, if we need atlases, transform them into DWI space
        if atlas_configs:
            warp_atlases_wf = init_warp_atlases_wf(atlas_configs=atlas_configs)
            workflow.connect([
                (inputnode, warp_atlases_wf, [('dwi_file', 'inputnode.source_file')]),
                (_get_source_node('template_to_acpc_xfm'), warp_atlases_wf, [
                    ('template_to_acpc_xfm', 'inputnode.template_to_acpc_xfm'),
                ]),
                (warp_atlases_wf, buffernode, [('outputnode.atlas_configs', 'atlas_configs')]),
            ])  # fmt:skip

    if 'mrtrix_5tt_hsvs' in extras_to_make and not has_qsiprep_5tt_hsvs:
        raise Exception('Unable to create a 5tt HSV image given input data.')

    # Directly connect anything from the inputs that we haven't created here
    workflow.connect([
        (inputnode, outputnode, [(name, name) for name in connect_from_inputnode]),
        (buffernode, outputnode, [(name, name) for name in connect_from_buffernode])
    ])  # fmt:skip

    workflow = clean_datasinks(workflow, qsirecon_suffix)
    return workflow, _get_status()


def init_warp_atlases_wf(atlas_configs):
    """Warp template-space atlases onto the DWI grid and convert them to MIF format.

    Each atlas is reoriented to LPS+, resampled onto the DWI grid with the
    template-to-AC-PC transform using nearest-neighbor (multi-label) interpolation,
    written to the derivatives with its label table, and converted to MRtrix3's MIF
    format for use in connectivity workflows.

    Parameters
    ----------
    atlas_configs : :obj:`dict`
        Atlas configurations from :func:`~qsirecon.utils.atlases.collect_atlases`,
        keyed by atlas name.
        Each entry provides the template-space ``image``, its ``labels`` table, and
        ``metadata``.

    Inputs
    ------
    source_file
        The preprocessed DWI, used both as the reference grid and to name the outputs.
    template_to_acpc_xfm
        Transform from template space to AC-PC space, as written by QSIPrep.

    Outputs
    -------
    atlas_configs
        ``atlas_configs`` with each entry updated to point to the DWI-space NIfTI and
        MIF images and the label tables that were written.
    """
    workflow = Workflow(name='warp_atlases_wf')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
                'template_to_acpc_xfm',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['atlas_configs'],
        ),
        name='outputnode',
    )

    atlas_str = describe_atlases(sorted(atlas_configs.keys()))
    workflow.__desc__ = (
        f'The following atlases were used in the workflow: {atlas_str}. '
        'Cortical parcellations were mapped from template space to DWIS '
        'using the spatial normalization of the anatomical image. '
    )

    # The atlas configs are known when the workflow is built, so the per-atlas files
    # can be collected into lists here and fed to MapNodes.
    atlas_names = sorted(atlas_configs.keys())
    atlas_images = [atlas_configs[name]['image'] for name in atlas_names]
    atlas_labels = [atlas_configs[name]['labels'] for name in atlas_names]

    # Reorient the atlases to LPS+ before resampling them
    conform_atlases = pe.MapNode(
        ConformAtlas(orientation='LPS'),
        iterfield=['in_file'],
        name='conform_atlases',
        run_without_submitting=True,
    )
    conform_atlases.inputs.in_file = atlas_images

    # Warp all atlases to source_file's space/resolution
    warp_atlases = pe.MapNode(
        ants.ApplyTransforms(interpolation='MultiLabel', dimension=3),
        iterfield=['input_image'],
        name='warp_atlases',
    )
    workflow.connect([
        (inputnode, warp_atlases, [
            ('source_file', 'reference_image'),
            ('template_to_acpc_xfm', 'transforms'),
        ]),
        (conform_atlases, warp_atlases, [('out_file', 'input_image')]),
    ])  # fmt:skip

    # Write out the lookup tables labelconvert needs to renumber each atlas's
    # indices into the sequential ones MRtrix's connectivity tools expect
    make_atlas_luts = pe.MapNode(
        AtlasLUTs(),
        iterfield=['atlas_labels_file'],
        name='make_atlas_luts',
        run_without_submitting=True,
    )
    make_atlas_luts.inputs.atlas_labels_file = atlas_labels

    # Renumber the atlas indices and convert the atlases to mif format
    convert_labels = pe.MapNode(
        mrtrix3.LabelConvert(),
        iterfield=['in_file', 'in_lut', 'in_config', 'out_file'],
        name='convert_labels',
    )
    convert_labels.inputs.out_file = [f'{name}_to_dwi.mif' for name in atlas_names]
    workflow.connect([
        (warp_atlases, convert_labels, [('output_image', 'in_file')]),
        (make_atlas_luts, convert_labels, [
            ('orig_lut', 'in_lut'),
            ('mrtrix_lut', 'in_config'),
        ]),
    ])  # fmt:skip

    ds_atlas = pe.MapNode(
        DerivativesDataSink(
            dismiss_entities=('desc',),
            suffix='dseg',
            compress=True,
        ),
        iterfield=['in_file', 'seg'],
        name='ds_atlas',
        run_without_submitting=True,
    )
    ds_atlas_mifs = pe.MapNode(
        DerivativesDataSink(
            dismiss_entities=('desc',),
            suffix='dseg',
            extension='.mif.gz',
            compress=True,
        ),
        iterfield=['in_file', 'seg'],
        name='ds_atlas_mifs',
        run_without_submitting=True,
    )
    ds_atlas_mrtrix_lut = pe.MapNode(
        DerivativesDataSink(
            dismiss_entities=('desc',),
            desc='mrtrix',
            suffix='dseg',
            extension='.txt',
        ),
        iterfield=['in_file', 'seg'],
        name='ds_atlas_mrtrix_lut',
        run_without_submitting=True,
    )
    ds_atlas_orig_lut = pe.MapNode(
        DerivativesDataSink(
            dismiss_entities=('desc',),
            suffix='dseg',
            extension='.txt',
        ),
        iterfield=['in_file', 'seg'],
        name='ds_atlas_orig_lut',
        run_without_submitting=True,
    )
    for ds_node in (ds_atlas, ds_atlas_mifs, ds_atlas_mrtrix_lut, ds_atlas_orig_lut):
        ds_node.inputs.seg = atlas_names

    workflow.connect([
        (warp_atlases, ds_atlas, [('output_image', 'in_file')]),
        (convert_labels, ds_atlas_mifs, [('out_file', 'in_file')]),
        (make_atlas_luts, ds_atlas_mrtrix_lut, [('mrtrix_lut', 'in_file')]),
        (make_atlas_luts, ds_atlas_orig_lut, [('orig_lut', 'in_file')]),
    ])  # fmt:skip

    # Collect the datasinked files back into the atlas configs, so that later nodes
    # that use the atlas configs reference the written-out files.
    recombine_atlas_configs = pe.Node(
        RecombineAtlasConfigs(atlas_configs=atlas_configs, atlases=atlas_names),
        name='recombine_atlas_configs',
        run_without_submitting=True,
    )
    workflow.connect([
        (ds_atlas, recombine_atlas_configs, [('out_file', 'nifti_files')]),
        (ds_atlas_mifs, recombine_atlas_configs, [('out_file', 'mif_files')]),
        (ds_atlas_mrtrix_lut, recombine_atlas_configs, [('out_file', 'mrtrix_lut_files')]),
        (ds_atlas_orig_lut, recombine_atlas_configs, [('out_file', 'orig_lut_files')]),
        (recombine_atlas_configs, outputnode, [('atlas_configs', 'atlas_configs')]),
    ])  # fmt:skip

    # Fill in the atlas datasinks
    for node in workflow.list_node_names():
        node_suffix = node.split('.')[-1]
        if node_suffix.startswith('ds_atlas'):
            workflow.connect([
                (inputnode, workflow.get_node(node), [('source_file', 'source_file')]),
            ])  # fmt:skip

    return workflow


def _get_first(item):
    if isinstance(item, (list, tuple)):
        return item[0]
    return item


def _get_resampled(atlas_configs, atlas_name, to_retrieve):
    return atlas_configs[atlas_name][to_retrieve]


def init_output_grid_wf() -> Workflow:
    """Generate a non-oblique, uniform voxel-size grid around a brain.

    The template image is cropped to its non-zero extent with a margin
    (AFNI's ``3dAutobox``), deobliqued, and resampled to the requested output
    resolution.
    When no ``--output-resolution`` was given, the voxel size is taken from the
    input image.

    Inputs
    ------
    template_image
        A brain-masked image in the space the grid should cover, typically the
        template image.
    input_image
        The image whose voxel size is used when no output resolution was requested,
        typically the DWI reference image.

    Outputs
    -------
    grid_image
        The reference grid, used as ``resampling_template`` by downstream nodes.

    Returns
    -------
    workflow : :obj:`niworkflows.engine.workflows.LiterateWorkflow`
        The workflow.
    """
    workflow = Workflow(name='output_grid_wf')

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['template_image', 'input_image']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['grid_image']), name='outputnode')
    # Create the output reference grid_image
    if config.workflow.output_resolution is None:
        voxel_size = traits.Undefined
    else:
        voxel_size = config.workflow.output_resolution
    padding = 4 if config.workflow.infant else 8

    autobox_template = pe.Node(
        afni.Autobox(outputtype='NIFTI_GZ', padding=padding), name='autobox_template'
    )
    deoblique_autobox = pe.Node(
        afni.Warp(outputtype='NIFTI_GZ', deoblique=True), name='deoblique_autobox'
    )
    voxel_size_chooser = pe.Node(
        VoxelSizeChooser(voxel_size=voxel_size), name='voxel_size_chooser'
    )
    resample_to_voxel_size = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ'), name='resample_to_voxel_size'
    )

    workflow.connect([
        (inputnode, autobox_template, [('template_image', 'in_file')]),
        (autobox_template, deoblique_autobox, [('out_file', 'in_file')]),
        (deoblique_autobox, resample_to_voxel_size, [('out_file', 'in_file')]),
        (resample_to_voxel_size, outputnode, [('out_file', 'grid_image')]),
        (inputnode, voxel_size_chooser, [('input_image', 'input_image')]),
        (voxel_size_chooser, resample_to_voxel_size, [(('voxel_size', _tupleize), 'voxel_size')])
    ])  # fmt:skip

    return workflow


def _tupleize(value):
    # Nipype did not like having a Tuple output trait
    return (value, value, value)
