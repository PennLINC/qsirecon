"""
Converting between file formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_mif_to_fibgz_wf
.. autofunction:: init_fibgz_to_mif_wf

"""

import logging

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ...interfaces.bids import DerivativesDataSink
from ...interfaces.converters import FODtoFIBGZ, MergeFODGQIFibs
from ...interfaces.images import ConformDwi
from ...interfaces.interchange import recon_workflow_input_fields
from ...utils.bids import clean_datasinks

LOGGER = logging.getLogger("nipype.workflow")


def init_mif_to_fibgz_wf(inputs_dict, name="mif_to_fibgz", qsirecon_suffix="", params={}):
    """Converts a MRTrix mif file to DSI Studio fib file.

    This workflow uses ``sh2amp`` to sample the FODs on the standard DSI Studio
    ODF direction set. These are then loaded and converted to the fib MATLAB v4 format
    and peak directions are detected using Dipy.

    Inputs

        mif_file
            MRTrix format mif file containing sh coefficients representing FODs.

    Outputs

        fibgz
            DSI Studio fib file containing the FODs from the input ``mif_file``.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["fod_sh_mif", "fibgz"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fibgz", "recon_scalars"]), name="outputnode"
    )
    outputnode.inputs.recon_scalars = []
    workflow = Workflow(name=name)
    convert_to_fib = pe.Node(FODtoFIBGZ(), name="convert_to_fib")
    workflow.connect([
        (inputnode, convert_to_fib, [('fod_sh_mif', 'mif_file')]),
        (convert_to_fib, outputnode, [('fib_file', 'fibgz')]),
    ])  # fmt:skip

    if qsirecon_suffix:
        # Save the output in the outputs directory
        ds_fibgz = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                extension=".fib.gz",
                compress=True,
            ),
            name="ds_fibgz",
            run_without_submitting=True,
        )
        workflow.connect(convert_to_fib, 'fib',
                         ds_fibgz, 'in_file')  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)


def init_fibgz_to_mif_wf(name="fibgz_to_mif", qsirecon_suffix="", params={}):
    """Needs Documentation"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["mif_file"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fib_file", "recon_scalars"]), name="outputnode"
    )
    outputnode.inputs.recon_scalars = []
    workflow = Workflow(name=name)
    convert_to_fib = pe.Node(FODtoFIBGZ(), name="convert_to_fib")
    workflow.connect([
        (inputnode, convert_to_fib, [('mif_file', 'mif_file')]),
        (convert_to_fib, outputnode, [('fib_file', 'fib_file')])
    ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)


def init_fod_fib_wf(inputs_dict, name="fod_fib", qsirecon_suffix="", params={}):
    """Converts MRTrix FODs to a DSI Studio fib file for the FOD-AutoTrack method.

    Inputs

        fod_sh_mif
            MRTrix format mif file containing sh coefficients representing FODs.
        fibgz
            DSI Studio file containing a standard GQI reconstruction
        fibgz_map
            DSI Studio spatial normalization file.

    Outputs

        fibgz
            DSI Studio fib file containing the FODs from the input ``mif_file``.
        fibgz_map
            DSI Studio spatial normalization file. Passed through unchanged, but renamed
            to match ``fibgz`` so it will be recognized by DSI Studio.
        recon_scalars
            Ununsed.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields + ["fod_sh_mif", "fibgz", "fibgz_map"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fibgz", "fibgz_map", "recon_scalars"]), name="outputnode"
    )
    outputnode.inputs.recon_scalars = []
    workflow = Workflow(name=name)

    # Convert the FOD sh nifti into a valid fib file
    convert_fod_to_fib = pe.Node(FODtoFIBGZ(), name="convert_fod_to_fib")
    # Merge the reference fib file and the FOD fib file
    merge_fod_and_qgi_fibs = pe.Node(MergeFODGQIFibs(), name="merge_fod_and_gqi_fibs")

    workflow.connect([
        (inputnode, convert_fod_to_fib, [
            ('fod_sh_mif', 'mif_file'),
            ('dwi_mask', 'mask_file'),
        ]),
        (inputnode, merge_fod_and_qgi_fibs, [
            ('fibgz', 'reference_fib_file'),
            ('fibgz_map', 'fibgz_map'),
        ]),
        (convert_fod_to_fib, merge_fod_and_qgi_fibs, [('fib_file', 'csd_fib_file')]),
        (merge_fod_and_qgi_fibs, outputnode, [
            ('fibgz_map', 'fibgz_map'),
            ('fibgz', 'fibgz'),
        ]),
    ])  # fmt:skip

    if qsirecon_suffix:
        # Save the output in the outputs directory
        ds_fibgz = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwimap",
                extension=".fib.gz",
                model=params.get("model", "csd"),
                compress=True,
            ),
            name="ds_fibgz",
            run_without_submitting=True,
        )
        workflow.connect(outputnode, "fibgz", ds_fibgz, "in_file")

    return clean_datasinks(workflow, qsirecon_suffix)


def init_qsirecon_to_fsl_wf(inputs_dict, name="qsirecon_to_fsl", qsirecon_suffix="", params={}):
    """Converts QSIRecon outputs (images, bval, bvec) to fsl standard orientation"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    to_reorient = ["mask_file", "dwi_file", "bval_file", "bvec_file", "recon_scalars"]
    outputnode = pe.Node(niu.IdentityInterface(fields=to_reorient), name="outputnode")
    workflow = Workflow(name=name)
    outputnode.inputs.recon_scalars = []

    convert_dwi_to_fsl = pe.Node(ConformDwi(orientation="LAS"), name="convert_to_fsl")
    convert_mask_to_fsl = pe.Node(ConformDwi(orientation="LAS"), name="convert_mask_to_fsl")
    workflow.connect([
        (inputnode, convert_dwi_to_fsl, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (convert_dwi_to_fsl, outputnode, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (inputnode, convert_mask_to_fsl, [('mask_file', 'dwi_file')]),
        (convert_mask_to_fsl, outputnode, [('dwi_file', 'mask_file')])
    ])  # fmt:skip

    if qsirecon_suffix:
        # Save the output in the outputs directory
        ds_dwi_file = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwi",
                extension=".nii.gz",
            ),
            name="ds_dwi_" + name,
            run_without_submitting=True,
        )
        ds_bval_file = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwi",
                extension=".bval",
            ),
            name="ds_bval_" + name,
            run_without_submitting=True,
        )
        ds_bvec_file = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwi",
                extension=".bvec",
            ),
            name="ds_bvec_" + name,
            run_without_submitting=True,
        )
        ds_mask_file = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="mask",
                extension=".nii.gz",
            ),
            name="ds_mask_" + name,
            run_without_submitting=True,
        )
        workflow.connect([
            (convert_dwi_to_fsl, ds_bval_file, [('bval_file', 'in_file')]),
            (convert_dwi_to_fsl, ds_bvec_file, [('bvec_file', 'in_file')]),
            (convert_dwi_to_fsl, ds_dwi_file, [('dwi_file', 'in_file')]),
            (convert_mask_to_fsl, ds_mask_file, [('dwi_file', 'in_file')])
        ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)
