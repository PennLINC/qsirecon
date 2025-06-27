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

from ...interfaces.bids import DerivativesDataSink
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import ReconScalarsTableSplitterDataSink
from ...interfaces.reports import ScalarReport
from ...interfaces.scalar_mapping import BundleMapper, TemplateMapper
from ...utils.bids import clean_datasinks
from .utils import init_scalar_output_wf

LOGGER = logging.getLogger("nipype.workflow")


def init_scalar_to_bundle_wf(inputs_dict, name="scalar_to_bundle", qsirecon_suffix="", params={}):
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
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields
            + ["tck_files", "bundle_names", "recon_scalars", "collected_scalars"],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["bundle_summary"]), name="outputnode")
    workflow = Workflow(name=name)
    bundle_mapper = pe.Node(BundleMapper(**params), name="bundle_mapper")
    ds_bundle_mapper = pe.Node(
        ReconScalarsTableSplitterDataSink(dismiss_entities=["desc"], suffix="scalarstats"),
        name="ds_bundle_mapper",
        run_without_submitting=True,
    )
    ds_tdi_summary = pe.Node(
        ReconScalarsTableSplitterDataSink(dismiss_entities=["desc"], suffix="tdistats"),
        name="ds_tdi_summary",
        run_without_submitting=True,
    )
    workflow.connect([
        (inputnode, bundle_mapper, [
            ("collected_scalars", "recon_scalars"),
            ("tck_files", "tck_files"),
            ("dwi_ref", "dwiref_image"),
            ("mapping_metadata", "mapping_metadata"),
            ("bundle_names", "bundle_names")]),
        (bundle_mapper, ds_bundle_mapper, [
            ("bundle_summary", "summary_tsv")]),
        (bundle_mapper, outputnode, [
            ("bundle_summary", "bundle_summary")]),
        (bundle_mapper, ds_tdi_summary, [
            ("tdi_stats", "summary_tsv")])
    ])  # fmt:skip

    # NOTE: Don't add qsirecon_suffix with clean_datasinks here,
    # as the qsirecon_suffix is determined within ReconScalarsTableSplitterDataSink.
    return clean_datasinks(workflow, qsirecon_suffix=None)


def init_scalar_to_atlas_wf(
    inputs_dict,
    name="scalar_to_template",
    qsirecon_suffix="",
    params={},
):
    """Map scalar images to atlas regions

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
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields + ["recon_scalars", "collected_scalars"]
        ),
        name="inputnode",
    )
    # outputnode = pe.Node(niu.IdentityInterface(fields=["atlas_summaries"]), name="outputnode")
    workflow = Workflow(name=name)
    bundle_mapper = pe.Node(BundleMapper(**params), name="bundle_mapper")
    workflow.connect([
        (inputnode, bundle_mapper, [
            ("collected_scalars", "recon_scalars"),
            ("tck_files", "tck_files"),
            ("dwi_ref", "dwiref_image")])
    ])  # fmt:skip
    if qsirecon_suffix:

        ds_bundle_summaries = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                desc="bundlemap",
            ),
            name="ds_bundle_summaries",
            run_without_submitting=True,
        )
        workflow.connect([
            (bundle_mapper, ds_bundle_summaries, [("bundle_summaries", "in_file")])
        ])  # fmt:skip
    return clean_datasinks(workflow, qsirecon_suffix)


def init_scalar_to_template_wf(
    inputs_dict,
    name="scalar_to_template",
    qsirecon_suffix="",
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
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields + ["recon_scalars", "collected_scalars"],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["template_scalars", "template_scalar_sidecars"]),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    template_mapper = pe.Node(
        TemplateMapper(template_space=inputs_dict["template_output_space"], **params),
        name="template_mapper",
    )

    scalar_output_wf = init_scalar_output_wf()
    workflow.connect([
        (inputnode, scalar_output_wf, [("dwi_file", "inputnode.source_file")]),
        (template_mapper, scalar_output_wf, [
            ("template_space_scalar_info", "inputnode.scalar_configs"),
            ("template_space", "inputnode.space"),
        ]),
    ])  # fmt:skip

    workflow.connect([
        (inputnode, template_mapper, [
            ("collected_scalars", "recon_scalars"),
            ("acpc_to_template_xfm", "to_template_transform"),
            ("resampling_template", "template_reference_image"),
        ]),
        (template_mapper, outputnode, [("template_space_scalars", "template_scalars")]),
    ])  # fmt:skip

    # Create a reportlet for the scalar maps
    scalar_report = pe.Node(
        ScalarReport(),
        name="scalar_report",
    )
    workflow.connect([
        (inputnode, scalar_report, [
            ("resampling_template", "underlay"),
            ("dwi_mask", "mask_file"),
            ("collected_scalars", "scalar_metadata"),
        ]),
        (outputnode, scalar_report, [("template_scalars", "scalar_maps")]),
    ])  # fmt:skip

    ds_scalars_figure = pe.Node(
        DerivativesDataSink(
            dismiss_entities=["desc"],
            datatype="dwi",
            desc="scalars",
            suffix="dwimap",
            extension="nii.gz",
        ),
        name="ds_scalars_figure",
        run_without_submitting=True,
    )
    workflow.connect([(scalar_report, ds_scalars_figure, [("out_report", "in_file")])])

    return clean_datasinks(workflow, qsirecon_suffix)


def init_scalar_to_surface_wf(
    inputs_dict,
    name="scalar_to_surface",
    qsirecon_suffix="",
    params={},
):
    """Maps scalar data to a surface."""
    raise NotImplementedError()
