#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import json
import os.path as op
import sys
from copy import deepcopy
from glob import glob

import nipype.pipeline.engine as pe
import yaml
from bids.layout import Query
from dipy import __version__ as dipy_ver
from nilearn import __version__ as nilearn_ver
from nipype import __version__ as nipype_ver
from nipype.interfaces import utility as niu
from nipype.utils.filemanip import split_filename
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.utils.misc import fix_multi_T1w_source_name
from packaging.version import Version
from pkg_resources import resource_filename as pkgrf

from .. import config
from ..interfaces.bids import CopyAtlas


def init_qsirecon_wf():
    """Organize the execution of qsirecon, with a sub-workflow for each subject."""
    ver = Version(config.environment.version)
    qsirecon_wf = Workflow(name=f"qsirecon_{ver.major}_{ver.minor}_wf")
    qsirecon_wf.base_dir = config.execution.work_dir

    if config.workflow.input_type not in ("qsiprep", "hcpya", "ukb"):
        raise NotImplementedError(
            f"{config.workflow.input_type} is not supported as recon-input yet."
        )

    to_recon_list = config.execution.participant_label

    for subject_id in to_recon_list:
        single_subject_wf = init_single_subject_recon_wf(subject_id=subject_id)

        single_subject_wf.config["execution"]["crashdump_dir"] = str(
            config.execution.output_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        qsirecon_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.output_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / "qsirecon.toml")

        # Dump a copy of the recon spec into the log directory as well
        recon_spec = _load_recon_spec(config.workflow.recon_spec)
        with open(log_dir / "recon_spec.yaml", "w") as f:
            yaml.dump(recon_spec, f, default_flow_style=False, sort_keys=False, indent=4)

    return qsirecon_wf


def init_single_subject_recon_wf(subject_id):
    """Organize the reconstruction pipeline for a single subject.

    Reconstruction is performed using a separate workflow for each dwi series.

    Parameters
    ----------
    subject_id : str
        Single subject label
    """
    from ..interfaces.bids import DerivativesDataSink
    from ..interfaces.ingress import QSIPrepDWIIngress
    from ..interfaces.interchange import (
        ReconWorkflowInputs,
        anatomical_workflow_outputs,
        qsiprep_output_names,
        recon_workflow_anatomical_input_fields,
        recon_workflow_input_fields,
    )
    from ..interfaces.reports import AboutSummary, SubjectSummary
    from ..interfaces.utils import GetUnique
    from ..utils.atlases import collect_atlases
    from ..utils.bids import collect_anatomical_data, get_entity
    from .recon.anatomical import (
        init_dwi_recon_anatomical_workflow,
        init_highres_recon_anatomical_wf,
    )
    from .recon.build_workflow import init_dwi_recon_workflow

    spec = _load_recon_spec(config.workflow.recon_spec)
    workflow = Workflow(name=f"sub-{subject_id}_{spec['name']}")

    workflow.__desc__ = f"""
Reconstruction was
performed using *QSIRecon* {config.__version__} (@cieslak2021qsiprep),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = f"""

Many internal operations of *QSIRecon* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362] and
*Dipy* {dipy_ver}[@dipy].
For more details of the pipeline, see [the section corresponding
to workflows in *QSIRecon*'s documentation]\
(https://qsirecon.readthedocs.io/en/latest/workflows.html).


### References

    """

    dwis_and_anats = _get_iterable_dwis_and_anats(subject_id)

    if len(dwis_and_anats) == 0:
        config.loggers.workflow.info("No dwi files found for %s", subject_id)
        return workflow

    anat_ingress_nodes = {}
    anat_input_files = [pair[1] for pair in dwis_and_anats]

    highres_anat_wfs = {}
    highres_anat_statuses = {}
    for anat_input_file in anat_input_files:
        _session_filter = (
            None
            if anat_input_file is None
            else anat_input_file.entities.get("session", Query.NONE)
        )

        # We only need to process unique anat files that match dwis
        if anat_input_file.path in highres_anat_wfs:
            continue
        anat_data, highres_anat_statuses[anat_input_file.path] = collect_anatomical_data(
            layout=config.execution.layout,
            subject_id=subject_id,
            session_id=_session_filter,
            needs_t1w_transform=bool(config.execution.atlases),
            bids_filters=config.execution.bids_filters or {},
        )
        config.loggers.workflow.info(
            f"Anatomical data available for {anat_input_file.path}:\n"
            f"{yaml.dump(anat_data, default_flow_style=False, indent=4)}"
        )
        anat_ingress_nodes[anat_input_file.path] = pe.Node(
            niu.IdentityInterface(fields=list(anat_data.keys())),
            name="anat_ingress_node",
        )
        highres_anat_wfs[anat_input_file.path], highres_anat_statuses[anat_input_file.path] = (
            init_highres_recon_anatomical_wf(
                subject_id=subject_id,
                session_id=_session_filter,
                extras_to_make=spec.get("anatomical", []),
                status=highres_anat_statuses[anat_input_file.path],
                anat_data=anat_data,
            )
        )
    config.loggers.workflow.info(f"Found {len(anat_input_files)} high-res anatomicals to process")

    atlas_configs = {}
    if config.execution.atlases:
        # Limit atlases to ones in the specified space.
        xfm_to_anat = anat_data["template_to_acpc_xfm"]
        template_space = get_entity(xfm_to_anat, "from")
        bids_filters = (config.execution.bids_filters or {}).copy()
        bids_filters["atlas"] = bids_filters.get("atlas", {})
        bids_filters["atlas"]["space"] = template_space

        # Collect atlases across datasets, including built-in atlases.
        atlas_configs = collect_atlases(
            datasets=config.execution.datasets,
            atlases=config.execution.atlases,
            bids_filters=bids_filters,
        )
        # Patch the transform into the atlas configs.
        # This is a placeholder until we can support atlases in various spaces.
        for atlas_name in atlas_configs.keys():
            atlas_configs[atlas_name]["xfm_to_anat"] = xfm_to_anat

        # Prepare the atlases.
        # Reorient to LPS+ and zero out the sform.
        for atlas_name, atlas_config in atlas_configs.items():
            # Node is named dataset_ instead of ds_ so no clean_datasinks step will affect it.
            # XXX: We should pass the outputs from these datasinks to any steps that use the
            # atlases in order to track Sources.
            ds_atlas_orig = pe.Node(
                CopyAtlas(
                    in_file=atlas_config["image"],
                    source_file=atlas_config["image"],
                    out_dir=config.execution.output_dir,
                    atlas=atlas_name,
                    meta_dict=atlas_config["metadata"],
                ),
                name=f"datasink_atlas_orig_{atlas_name}",
            )
            workflow.add_nodes([ds_atlas_orig])

            ds_atlas_labels_orig = pe.Node(
                CopyAtlas(
                    in_file=atlas_config["labels"],
                    source_file=atlas_config["labels"],
                    out_dir=config.execution.output_dir,
                    atlas=atlas_name,
                ),
                name=f"datasink_atlas_labels_orig_{atlas_name}",
            )
            workflow.add_nodes([ds_atlas_labels_orig])

    # create a processing pipeline for the dwis in each session
    dwi_recon_wfs = {}
    dwi_anat_wfs = {}
    recon_full_inputs = {}
    dwi_ingress_nodes = {}
    anat_ingress_wfs = {}
    dwi_files = [dwi_input["bids_dwi_file"] for dwi_input in dwis_and_anats]
    for dwi_input, anat_input in enumerate(dwis_and_anat_wfs):
        dwi_file = dwi_input["bids_dwi_file"]
        wf_name = _get_wf_name(dwi_file)

        # Get the preprocessed DWI and all the related preprocessed images
        dwi_ingress_nodes[dwi_file] = pe.Node(
            QSIPrepDWIIngress(dwi_file=dwi_file),
            name=f"{wf_name}_ingressed_dwi_data",
        )

        # Create scan-specific anatomical data (mask, atlas configs, odf ROIs for reports)
        dwi_anat_wfs[dwi_file], dwi_available_anatomical_data = init_dwi_recon_anatomical_workflow(
            atlas_configs=atlas_configs,
            prefer_dwi_mask=False,
            needs_t1w_transform=bool(config.execution.atlases),
            extras_to_make=spec.get("anatomical", []),
            name=f"{wf_name}_dwi_specific_anat_wf",
            **highres_anat_statuses[anat_input],
        )

        # This node holds all the inputs that will go to the recon workflow.
        # It is the definitive place to check what the input files are
        recon_full_inputs[dwi_file] = pe.Node(
            ReconWorkflowInputs(),
            name=f"{wf_name}_recon_inputs",
        )

        # This is the actual recon workflow for this dwi file
        dwi_recon_wfs[dwi_file] = init_dwi_recon_workflow(
            available_anatomical_data=dwi_available_anatomical_data,
            workflow_spec=spec,
            name=f"{wf_name}_recon_wf",
        )

        # Connect the collected diffusion data (gradients, etc) to the inputnode
        workflow.connect([
            # The dwi data
            (dwi_ingress_nodes[dwi_file], recon_full_inputs[dwi_file], [
                (trait, trait) for trait in qsiprep_output_names
            ]),

            # Session-specific anatomical data
            (dwi_ingress_nodes[dwi_file], dwi_anat_wfs[dwi_file], [
                (trait, f"inputnode.{trait}") for trait in qsiprep_output_names
            ]),

            # subject dwi-specific anatomical to a special node in recon_full_inputs so
            # we have a record of what went in. Otherwise it would be lost in an IdentityInterface
            (dwi_anat_wfs[dwi_file], recon_full_inputs[dwi_file], [
                (f"outputnode.{trait}", trait) for trait in recon_workflow_anatomical_input_fields
            ]),

            # send the recon_full_inputs to the dwi recon workflow
            (recon_full_inputs[dwi_file], dwi_recon_wfs[dwi_file], [
                (trait, f"inputnode.{trait}") for trait in recon_workflow_input_fields
            ]),

            (anat_ingress_wfs[dwi_file], dwi_anat_wfs[dwi_file], [
                (f"outputnode.{trait}", f"inputnode.{trait}")
                for trait in anatomical_workflow_outputs
            ]),
        ])  # fmt:skip

    about = pe.Node(
        AboutSummary(
            version=config.environment.version,
            command=" ".join(sys.argv),
        ),
        name="about",
        run_without_submitting=True,
    )
    summary = pe.Node(
        SubjectSummary(
            subject_id=subject_id,
            subjects_dir=config.execution.fs_subjects_dir,
            std_spaces=["MNIInfant" if config.workflow.infant else "MNI152NLin2009cAsym"],
            nstd_spaces=[],
            dwi=dwi_files,
        ),
        name="summary",
        run_without_submitting=True,
    )
    workflow.connect([(reduce_t1_preproc, summary, [("outlist", "t1w")])])

    suffix_dirs = []
    for qsirecon_suffix in config.workflow.qsirecon_suffixes:
        suffix_dir = str(
            config.execution.output_dir / "derivatives" / f"qsirecon-{qsirecon_suffix}"
        )
        suffix_dirs.append(suffix_dir)

    ds_report_about = pe.MapNode(
        DerivativesDataSink(
            source_file=dwi_basename,
            datatype="figures",
            desc="about",
            suffix="T1w",
        ),
        name="ds_report_about",
        run_without_submitting=True,
        iterfield=["base_directory"],
    )
    ds_report_about.inputs.base_directory = suffix_dirs
    workflow.connect([(about, ds_report_about, [("out_report", "in_file")])])

    ds_report_summary = pe.MapNode(
        DerivativesDataSink(
            source_file=dwi_basename,
            datatype="figures",
            desc="summary",
            suffix="T1w",
        ),
        name="ds_report_summary",
        run_without_submitting=True,
        iterfield=["base_directory"],
    )
    ds_report_summary.inputs.base_directory = suffix_dirs
    workflow.connect([(summary, ds_report_summary, [("out_report", "in_file")])])

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_report"):
            workflow.get_node(node).inputs.datatype = "figures"

    return workflow


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[:-1]).replace("-", "_")


def _load_recon_spec(spec_name):
    from copy import deepcopy

    from ..utils.misc import load_yaml
    from ..utils.sloppy_recon import make_sloppy

    prepackaged_dir = pkgrf("qsirecon", "data/pipelines")
    prepackaged = [op.split(fname)[1][:-5] for fname in glob(op.join(prepackaged_dir, "*.yaml"))]
    if op.exists(spec_name):
        recon_spec = spec_name
    elif spec_name in prepackaged:
        recon_spec = op.join(prepackaged_dir, f"{spec_name}.yaml")
    else:
        raise Exception(f"{spec_name} is not a file that exists or in {prepackaged}")

    if recon_spec.endswith(".json"):
        with open(recon_spec, "r") as f:
            try:
                spec = json.load(f)
            except Exception:
                raise Exception("Unable to read JSON spec. Check the syntax.")
    else:
        try:
            spec = load_yaml(recon_spec)
        except Exception:
            raise Exception("Unable to read YAML spec. Check the syntax.")

    if config.execution.sloppy:
        config.loggers.workflow.warning("Forcing reconstruction to use unrealistic parameters")
        spec = make_sloppy(spec)

    # Expand any "scalars_from" lists into separate nodes
    orig_spec = deepcopy(spec)
    spec["nodes"] = []
    for node in orig_spec["nodes"]:
        if "scalars_from" in node.keys() and isinstance(node["scalars_from"], list):
            for scalar_source in node["scalars_from"]:
                new_node = node.copy()
                new_node["name"] = f"{node['name']}_{scalar_source}"
                new_node["scalars_from"] = scalar_source
                if "qsirecon_suffix" not in new_node.keys():
                    # Infer the suffix from the source node
                    for temp_node in spec["nodes"]:
                        if temp_node["name"] == scalar_source:
                            new_node["qsirecon_suffix"] = temp_node["qsirecon_suffix"]
                            continue

                spec["nodes"].append(new_node)
        else:
            spec["nodes"].append(node)

    return spec


def _get_iterable_dwis_and_anats(subject_id):
    """Look through the BIDS Layout for DWIs and their corresponding anats.

    Parameters:
    -----------
    subject_id: str
        Subject label (without "sub-") to find data for

    Returns:
    --------
    dwis_and_anats: list of tuple


    """

    dwis_and_anats = []
    dwi_files = config.execution.layout.get(
        suffix="dwi",
        session=Query.OPTIONAL,
        space="T1w",
        extension=["nii", "nii.gz"],
    )

    for dwi_scan in dwi_files:
        subject_level_anats = config.execution.layout.get(
            suffix=["T1w", "T2w"],
            session=Query.NONE,
            space=Query.NONE,
            extension=["nii", "nii.gz"],
        )

        session_level_anats = []
        if dwi_session := dwi_scan.entities.get("session"):
            session_level_anats = config.execution.layout.get(
                suffix=["T1w", "T2w"],
                session=dwi_session,
                space=Query.NONE,
                extension=["nii", "nii.gz"],
            )

        if not (session_level_anats or subject_level_anats):
            anat_scan = None
        else:
            best_anat_source = session_level_anats if session_level_anats else subject_level_anats
            anat_scan = best_anat_source[0]

        dwis_and_anats.append((dwi_scan, anat_scan))
    return dwis_and_anats
