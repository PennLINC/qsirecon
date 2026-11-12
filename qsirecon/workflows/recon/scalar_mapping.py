"""
Summarize and Transform recon outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import logging
import re

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import afni
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from qsirecon import config

from ...interfaces.anatomical import GetTemplate
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import (
    ParcellateScalars,
    ParcellationTableSplitterDataSink,
    ReconScalarsTableSplitterDataSink,
)
from ...interfaces.scalar_mapping import BundleMapper, TemplateMapper
from ...interfaces.utils import SplitAtlasConfigs
from ...utils.bids import clean_datasinks
from ...utils.spaces import SpaceSpec, format_zooms
from .anatomical import init_output_grid_wf
from .utils import init_scalar_output_wf

LOGGER = logging.getLogger('nipype.workflow')


def init_scalar_to_bundle_wf(inputs_dict, name='scalar_to_bundle', qsirecon_suffix='', params={}):
    """Map scalar images to bundles

    Inputs
        tck_files
            MRtrix3 format tck files for each bundle
        bundle_names
            Names that describe which bundles are present in `tck_files`
        recon_scalars
            List of dictionaries containing scalar info

    Outputs

        bundle_summaries
            summary statistics in tsv format

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = 'Scalar NIfTI files were mapped to bundles.'

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields
            + ['tck_files', 'bundle_names', 'recon_scalars', 'collected_scalars'],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['bundle_summary']), name='outputnode')

    bundle_mapper = pe.Node(BundleMapper(**params), name='bundle_mapper')
    ds_bundle_mapper = pe.Node(
        ReconScalarsTableSplitterDataSink(dismiss_entities=['desc'], suffix='scalarstats'),
        name='ds_bundle_mapper',
        run_without_submitting=True,
    )
    ds_tdi_summary = pe.Node(
        ReconScalarsTableSplitterDataSink(dismiss_entities=['desc'], suffix='tdistats'),
        name='ds_tdi_summary',
        run_without_submitting=True,
    )
    workflow.connect([
        (inputnode, bundle_mapper, [
            ('collected_scalars', 'recon_scalars'),
            ('tck_files', 'tck_files'),
            ('dwi_ref', 'dwiref_image'),
            ('mapping_metadata', 'mapping_metadata'),
            ('bundle_names', 'bundle_names')]),
        (bundle_mapper, ds_bundle_mapper, [
            ('bundle_summary', 'summary_tsv')]),
        (bundle_mapper, outputnode, [
            ('bundle_summary', 'bundle_summary')]),
        (bundle_mapper, ds_tdi_summary, [
            ('tdi_stats', 'summary_tsv')])
    ])  # fmt:skip

    # NOTE: Don't add qsirecon_suffix with clean_datasinks here,
    # as the qsirecon_suffix is determined within ReconScalarsTableSplitterDataSink.
    return clean_datasinks(workflow, qsirecon_suffix=None)


def init_scalar_to_atlas_wf(
    inputs_dict,
    name='scalar_to_atlas_wf',
    qsirecon_suffix='',
    params={},
):
    """Parcellate scalar images using atlases.

    The atlases will be in the T1w space of the DWI data, produced by the atlas nodes in
    :func:`~qsirecon.workflows.recon.anatomical.init_dwi_recon_anatomical_workflow`.

    Inputs
        recon_scalars
            List of dictionaries containing scalar info. Unused.
        collected_scalars
            List of dictionaries containing scalar info. Used.
        atlas_configs
            Dictionary containing atlas configuration information.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = 'Scalar NIfTI files were parcellated using atlases.'

    input_fields = recon_workflow_input_fields + [
        'recon_scalars',
        'collected_scalars',
        'atlas_configs',
    ]
    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_fields),
        name='inputnode',
    )

    split_atlas_configs = pe.Node(
        SplitAtlasConfigs(),
        name='split_atlas_configs',
    )
    workflow.connect([(inputnode, split_atlas_configs, [('atlas_configs', 'atlas_configs')])])

    # Parcellates all scalars with one atlas at a time.
    # Outputs a tsv of parcellated scalar stats and atlas name ("seg").
    # Also ingresses and outputs metadata.
    scalar_parcellator = pe.MapNode(
        ParcellateScalars(**params),
        name='scalar_parcellator',
        iterfield=['atlas_config'],
    )
    workflow.connect([
        (inputnode, scalar_parcellator, [
            ('collected_scalars', 'scalars_config'),
            ('mapping_metadata', 'mapping_metadata'),
            ('dwi_mask', 'brain_mask'),
        ]),
        (split_atlas_configs, scalar_parcellator, [('atlas_configs', 'atlas_config')]),
    ])  # fmt:skip

    ds_parcellated_scalars = pe.MapNode(
        ParcellationTableSplitterDataSink(
            dataset_links=config.execution.dataset_links,
            dismiss_entities=['desc'],
            suffix='scalarstats',
        ),
        name='ds_parcellated_scalars',
        run_without_submitting=True,
        iterfield=['seg', 'in_file', 'meta_dict'],
    )
    workflow.connect([
        (scalar_parcellator, ds_parcellated_scalars, [
            ('parcellated_scalar_tsv', 'in_file'),
            ('metadata', 'meta_dict'),
            ('seg', 'seg'),
        ]),
    ])  # fmt:skip

    # NOTE: Don't add qsirecon_suffix with clean_datasinks here,
    # as the qsirecon_suffix is determined within ParcellationTableSplitterDataSink.
    return clean_datasinks(workflow, qsirecon_suffix=None)


def init_scalar_to_template_wf(
    inputs_dict,
    name='scalar_to_template',
    qsirecon_suffix='',
    params={},
):
    """Map scalar data to every volumetric template space requested with ``--output-spaces``.

    One branch is built per requested output space and resolution
    (the ``output_space_xfms`` entries of ``inputs_dict``, from
    :func:`~qsirecon.utils.bids.resolve_output_space_transforms`):

    1. the template is fetched from TemplateFlow, brain-masked and reoriented to LPS+
       (shared by the branches that use the same template at the same TemplateFlow
       resolution);
    2. a reference grid is built around it at the requested voxel size
       (:func:`~qsirecon.workflows.recon.anatomical.init_output_grid_wf`);
    3. the scalar maps are resampled onto that grid with the transforms for the space,
       which are known when the workflow is built and set directly on the node; and
    4. the resampled maps are written out with the ``space`` entity of the template and,
       when a resolution was requested for the space, its ``res`` entity.

    Nodes are named after the output space, for example
    ``template_mapper_MNI152NLin2009cAsym_res_2mm``.

    Inputs
        recon_scalars
            List of dictionaries containing scalar info
        dwi_ref
            The DWI reference image, whose voxel size is used for the output spaces
            that were requested without a resolution.

    Outputs

        template_scalars
            List of transformed files, for every output space
        template_recon_scalars
            Scalar configs for the transformed files, for every output space

    """
    output_space_xfms = inputs_dict.get('output_space_xfms') or []
    if not inputs_dict.get('has_qsiprep_anat_transforms') or not output_space_xfms:
        raise ValueError(
            f'The "{name}" node (action: template_map) requires ACPC-to-template transforms '
            'from the preprocessed anatomical data, but none were found for '
            f'{inputs_dict.get("dwi_file", "this DWI")}. '
            'This happens when the input dataset was preprocessed with --anat-modality none, '
            'or when its anatomical-to-template transform files are missing.'
        )

    workflow = Workflow(name=name)
    workflow.__desc__ = _describe_template_mapping(output_space_xfms)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields + ['recon_scalars', 'collected_scalars'],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'template_scalars',
                'template_scalar_sidecars',
                # scalar configs for the template spaces
                'template_recon_scalars',
            ],
        ),
        name='outputnode',
    )

    n_spaces = len(output_space_xfms)
    # The outputs of every space are gathered into one list each
    merge_scalars = pe.Node(niu.Merge(n_spaces), name='merge_template_scalars')
    merge_configs = pe.Node(niu.Merge(n_spaces), name='merge_template_recon_scalars')
    workflow.connect([
        (merge_scalars, outputnode, [('out', 'template_scalars')]),
        (merge_configs, outputnode, [('out', 'template_recon_scalars')]),
    ])  # fmt:skip

    # Brain-masked, LPS+ templates, keyed by template and TemplateFlow resolution.
    # Resolutions given as a physical size all start from the same template image.
    templates = {}
    for i_space, space_xfms in enumerate(output_space_xfms, start=1):
        output_space = SpaceSpec.from_token(space_xfms['token'])
        label = output_space.node_label

        template_key = (space_xfms['space'], output_space.templateflow_resolution)
        if template_key not in templates:
            templates[template_key] = _init_template_nodes(
                workflow,
                template_name=space_xfms['space'],
                resolution=output_space.templateflow_resolution,
            )

        output_grid_wf = init_output_grid_wf(
            output_space=output_space,
            name=f'output_grid_{label}_wf',
        )
        template_mapper = pe.Node(
            TemplateMapper(
                template_space=space_xfms['space'],
                to_template_transform=space_xfms['acpc_to_template_xfm'],
                **params,
            ),
            name=f'template_mapper_{label}',
        )
        scalar_output_wf = init_scalar_output_wf(
            name=f'scalar_output_{label}_wf',
            res=output_space.res_label,
        )
        workflow.connect([
            (inputnode, output_grid_wf, [('dwi_ref', 'inputnode.input_image')]),
            (templates[template_key], output_grid_wf, [('out_file', 'inputnode.template_image')]),
            (inputnode, template_mapper, [('collected_scalars', 'recon_scalars')]),
            (output_grid_wf, template_mapper, [
                ('outputnode.grid_image', 'template_reference_image'),
            ]),
            (inputnode, scalar_output_wf, [('dwi_file', 'inputnode.source_file')]),
            (template_mapper, scalar_output_wf, [
                ('template_space_scalar_info', 'inputnode.scalar_configs'),
                ('template_space', 'inputnode.space'),
            ]),
            (template_mapper, merge_scalars, [('template_space_scalars', f'in{i_space}')]),
            (scalar_output_wf, merge_configs, [('outputnode.scalar_configs', f'in{i_space}')]),
        ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)


def _init_template_nodes(workflow, template_name, resolution):
    """Add the nodes that fetch a template, mask it and reorient it to LPS+.

    Parameters
    ----------
    workflow : :obj:`niworkflows.engine.workflows.LiterateWorkflow`
        The workflow to add the nodes to.
    template_name : :obj:`str`
        TemplateFlow template name, with any cohort appended as ``+<cohort>``.
    resolution : :obj:`str`
        TemplateFlow resolution label to fetch the template at.

    Returns
    -------
    reorient_to_lps : :obj:`nipype.pipeline.engine.Node`
        The node whose ``out_file`` is the brain-masked template in LPS+ orientation.
    """
    label = re.sub(r'[^A-Za-z0-9]+', '_', f'{template_name}_tfres-{resolution}')
    get_template = pe.Node(
        GetTemplate(template_name=template_name, resolution=resolution),
        name=f'get_template_{label}',
    )
    mask_template = pe.Node(
        afni.Calc(expr='a*b', outputtype='NIFTI_GZ'),
        name=f'mask_template_{label}',
    )
    reorient_to_lps = pe.Node(
        afni.Resample(orientation='RAI', outputtype='NIFTI_GZ'),
        name=f'reorient_to_lps_{label}',
    )
    workflow.connect([
        (get_template, mask_template, [
            ('template_file', 'in_file_a'),
            ('mask_file', 'in_file_b'),
        ]),
        (mask_template, reorient_to_lps, [('out_file', 'in_file')]),
    ])  # fmt:skip
    return reorient_to_lps


def _describe_grid(output_space):
    """Describe the voxel size of an output space for the methods boilerplate."""
    resolution = output_space.resolution
    if resolution is None:
        return 'at the smallest voxel dimension of the preprocessed DWI'

    if resolution.kind == 'label':
        return f"on the template's res-{resolution.label} grid"

    return f'with {format_zooms(resolution.zooms)} voxels'


def _describe_template_mapping(output_space_xfms):
    """Write the methods boilerplate for the template-space scalar maps.

    Every requested output space and resolution is named, along with the templates
    that any chain of transforms goes through.
    """
    grids = {}
    chains = {}
    for space_xfms in output_space_xfms:
        space = space_xfms['space']
        output_space = SpaceSpec.from_token(space_xfms['token'])
        grids.setdefault(space, []).append(_describe_grid(output_space))
        if space_xfms.get('transform_path'):
            chains[space] = space_xfms['transform_path']

    if len(grids) == 1:
        space, space_grids = next(iter(grids.items()))
        desc = (
            f'Scalar NIfTI files were warped to {space} template space, '
            f'{" and ".join(space_grids)}.'
        )
    else:
        described = [
            f'{space} ({" and ".join(space_grids)})' for space, space_grids in grids.items()
        ]
        desc = (
            f'Scalar NIfTI files were warped to {len(grids)} template spaces: '
            f'{"; ".join(described)}.'
        )

    for space, path in chains.items():
        via = ', '.join(path[1:-1])
        desc += (
            f' To reach {space}, the transform from preprocessing to {via} was combined '
            'with template-to-template transforms from TemplateFlow.'
        )

    return desc


def init_scalar_to_surface_wf(
    inputs_dict,
    name='scalar_to_surface',
    qsirecon_suffix='',
    params={},
):
    """Maps scalar data to a surface."""
    raise NotImplementedError()
