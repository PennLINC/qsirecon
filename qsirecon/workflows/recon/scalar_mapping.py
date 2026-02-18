"""
Summarize and Transform recon outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_scalar_to_bundle_wf
.. autofunction:: init_scalar_to_atlas_wf
.. autofunction:: init_scalar_to_template_wf
.. autofunction:: init_scalar_to_surface_wf


"""

import logging

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import (
    ParcellateScalars,
    ParcellationTableSplitterDataSink,
    ReconScalarsTableSplitterDataSink,
)
from ...interfaces.scalar_mapping import BundleMapper, TemplateMapper
from ...interfaces.utils import SplitAtlasConfigs
from ...utils.bids import clean_datasinks
from .utils import init_scalar_output_wf
from qsirecon import config

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

    The atlases will be in the T1w space of the DWI data, produced by WarpConnectivityAtlases.

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
    """Maps scalar data to a volumetric template


    Inputs
        recon_scalars
            List of dictionaries containing scalar info

    Outputs

        template_scalars
            List of transformed files

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = (
        f'Scalar NIfTI files were warped to {inputs_dict["template_output_space"]} template space.'
    )

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
                # scalar configs for the template space
                'template_recon_scalars',
            ],
        ),
        name='outputnode',
    )

    template_mapper = pe.Node(
        TemplateMapper(template_space=inputs_dict['template_output_space'], **params),
        name='template_mapper',
    )
    workflow.connect([
        (inputnode, template_mapper, [
            ('collected_scalars', 'recon_scalars'),
            ('acpc_to_template_xfm', 'to_template_transform'),
            ('resampling_template', 'template_reference_image'),
        ]),
        (template_mapper, outputnode, [('template_space_scalars', 'template_scalars')]),
    ])  # fmt:skip

    scalar_output_wf = init_scalar_output_wf()
    workflow.connect([
        (inputnode, scalar_output_wf, [('dwi_file', 'inputnode.source_file')]),
        (template_mapper, scalar_output_wf, [
            ('template_space_scalar_info', 'inputnode.scalar_configs'),
            ('template_space', 'inputnode.space'),
        ]),
        (scalar_output_wf, outputnode, [
            ('outputnode.scalar_configs', 'template_recon_scalars'),
        ]),
    ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)


def init_scalar_to_surface_wf(
    inputs_dict,
    name='scalar_to_surface',
    qsirecon_suffix='',
    params={},
):
    """Maps scalar data to a surface."""
    raise NotImplementedError()
