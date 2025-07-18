"""
DSI Studio workflows
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dsi_studio_recon_wf
.. autofunction:: init_dsi_studio_connectivity_wf
.. autofunction:: init_dsi_studio_export_wf
.. autofunction:: init_dsi_studio_autotrack_wf
.. autofunction:: init_dsi_studio_connectivity_wf
.. autofunction:: init_dsi_studio_export_wf


"""

import logging

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.converters import DSIStudioTrkToTck
from ...interfaces.dsi_studio import (
    DSI_STUDIO_VERSION,
    AggregateAutoTrackResults,
    AutoTrack,
    ChenAutoTrack,
    DSIStudioAtlasGraph,
    DSIStudioCreateSrc,
    DSIStudioExport,
    DSIStudioGQIReconstruction,
    DSIStudioTracking,
    FixDSIStudioExportHeader,
    _get_dsi_studio_bundles,
)
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import DSIStudioReconScalars
from ...interfaces.reports import CLIReconPeaksReport, ConnectivityReport, ScalarReport
from ...utils.bids import clean_datasinks
from ...utils.misc import remove_non_alphanumeric
from .utils import init_scalar_output_wf

LOGGER = logging.getLogger("nipype.interface")


def init_dsi_studio_recon_wf(inputs_dict, name="dsi_studio_recon", qsirecon_suffix="", params={}):
    """Reconstructs diffusion data using DSI Studio.

    This workflow creates a ``.src.gz`` file from the input dwi, bvals and bvecs,
    then reconstructs ODFs using GQI.

    Inputs

        *Default qsirecon inputs*

    Outputs

        fibgz
            A DSI Studio fib file containing GQI ODFs, peaks and scalar values.

    Params

        ratio_of_mean_diffusion_distance: float
            Default 1.25. Distance to sample EAP at.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["odf_rois"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fibgz", "recon_scalars"]), name="outputnode"
    )
    workflow = Workflow(name=name)
    outputnode.inputs.recon_scalars = []
    plot_reports = not config.execution.skip_odf_reports
    omp_nthreads = config.nipype.omp_nthreads
    desc = """DSI Studio Reconstruction

: """
    create_src = pe.Node(DSIStudioCreateSrc(), name="create_src")
    romdd = params.get("ratio_of_mean_diffusion_distance", 1.25)
    gqi_recon = pe.Node(
        DSIStudioGQIReconstruction(ratio_of_mean_diffusion_distance=romdd),
        name="gqi_recon",
        n_procs=omp_nthreads,
    )
    desc += """\
Diffusion orientation distribution functions (ODFs) were reconstructed using
generalized q-sampling imaging (GQI, @yeh2010gqi) with a ratio of mean diffusion
distance of %02f in DSI Studio (version %s). """ % (
        romdd,
        DSI_STUDIO_VERSION,
    )

    workflow.connect([
        (inputnode, create_src, [
            ('dwi_file', 'input_nifti_file'),
            ('bval_file', 'input_bvals_file'),
            ('bvec_file', 'input_bvecs_file')]),
        (create_src, gqi_recon, [('output_src', 'input_src_file')]),
        (inputnode, gqi_recon, [('dwi_mask', 'mask')]),
        (gqi_recon, outputnode, [('output_fib', 'fibgz')])
    ])  # fmt:skip
    if plot_reports:
        # Make a visual report of the model
        plot_peaks = pe.Node(
            CLIReconPeaksReport(subtract_iso=True),
            name="plot_peaks",
            n_procs=omp_nthreads,
        )
        ds_report_peaks = pe.Node(
            DerivativesDataSink(
                desc="GQIODF",
                suffix="peaks",
                extension=".png",
            ),
            name="ds_report_peaks",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, plot_peaks, [
                ('dwi_ref', 'background_image'),
                ('odf_rois', 'odf_rois'),
                ('dwi_mask', 'mask_file'),
            ]),
            (gqi_recon, plot_peaks, [('output_fib', 'fib_file')]),
            (plot_peaks, ds_report_peaks, [('peak_report', 'in_file')]),
        ])  # fmt:skip

        # Plot targeted regions
        if inputs_dict["has_qsiprep_t1w_transforms"]:
            ds_report_odfs = pe.Node(
                DerivativesDataSink(
                    desc="GQIODF",
                    suffix="odfs",
                    extension=".png",
                ),
                name="ds_report_odfs",
                run_without_submitting=True,
            )
            workflow.connect([(plot_peaks, ds_report_odfs, [("odf_report", "in_file")])])

    if qsirecon_suffix:
        # Save the output in the outputs directory
        ds_gqi_fibgz = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwimap",
                extension=".fib.gz",
                model="gqi",
                compress=True,
            ),
            name="ds_gqi_fibgz",
            run_without_submitting=True,
        )
        workflow.connect(gqi_recon, 'output_fib',
                         ds_gqi_fibgz, 'in_file')  # fmt:skip
    workflow.__desc__ = desc

    return clean_datasinks(workflow, qsirecon_suffix)


def init_dsi_studio_tractography_wf(
    inputs_dict,
    name="dsi_studio_tractography",
    params={},
    qsirecon_suffix="",
):
    """Calculate streamline-based connectivity matrices using DSI Studio.

    DSI Studio has a deterministic tractography algorithm that can be used to
    estimate pairwise regional connectivity. It calculates multiple connectivity
    measures.

    Inputs

        fibgz
            A DSI Studio fib file produced by DSI Studio reconstruction.
        trk_file
            a DSI Studio trk.gz file

    Outputs

        trk_file
            A DSI-Studio format trk file
        fibgz
            The input fib file, as it is needed by downstream nodes in addition to
            the trk file.

    Params

        fiber_count
            number of streamlines to generate. Cannot also specify seed_count
        seed_count
            Number of seeds to track from. Does not guarantee a fixed number of
            streamlines and cannot be used with the fiber_count option.
        method
            0: streamline (Euler) 4: Runge Kutta
        seed_plan
            0: = traits.Enum((0, 1), argstr="--seed_plan=%d")
        initial_dir
            Seeds begin oriented as 0: the primary orientation of the ODF 1: a random orientation
            or 2: all orientations
        connectivity_type
            "pass" to count streamlines passing through a region. "end" to force
            streamlines to terminate in regions they count as connecting.
        connectivity_value
            "count", "ncount", "fa" used to quantify connection strength.
        random_seed
            Setting to True generates truly random (not-reproducible) seeding.
        fa_threshold
            If not specified, will use the DSI Studio Otsu threshold. Otherwise
            specigies the minimum qa value per fixed to be used for tracking.
        step_size
            Streamline propagation step size in millimeters.
        turning_angle
            Maximum turning angle in degrees for steamline propagation.
        smoothing
            DSI Studio smoothing factor
        min_length
            Minimum streamline length in millimeters.
        max_length
            Maximum streamline length in millimeters.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["fibgz"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["trk_file", "fibgz", "recon_scalars"]), name="outputnode"
    )
    outputnode.inputs.recon_scalars = []
    omp_nthreads = config.nipype.omp_nthreads
    workflow = Workflow(name=name)
    desc = (
        "#### DSI Studio Tractography\n\nTractography was run in DSI Studio "
        "(version %s) using a deterministic algorithm "
        "[@yeh2013deterministic]. " % DSI_STUDIO_VERSION
    )
    tracking = pe.Node(
        DSIStudioTracking(num_threads=omp_nthreads, **params),
        name="tracking",
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (inputnode, tracking, [('fibgz', 'input_fib')]),
        (tracking, outputnode, [('output_trk', 'trk_file')]),
        (inputnode, outputnode, [('fibgz', 'fibgz')])
    ])  # fmt:skip
    if qsirecon_suffix:
        # Save the output in the outputs directory
        ds_tracking = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                model="qathresh",
                suffix="streamlines",
            ),
            name="ds_" + name,
            run_without_submitting=True,
        )
        workflow.connect(tracking, 'output_trk',
                         ds_tracking, 'in_file')  # fmt:skip
    workflow.__desc__ = desc

    return clean_datasinks(workflow, qsirecon_suffix)


def init_dsi_studio_autotrack_registration_wf(
    inputs_dict,
    params={},
    qsirecon_suffix="",
    name="dsi_studio_autotrack_registration_wf",
):
    """Run DSI Studio's AutoTrack method to create a map.gz file. No bundles are saved.

    This workflow is designed to be used as input to other workflows that need both a
    fib and a map file (eg FOD Autotrack). The registration is better using GQI scalars
    than the imported FOD scalars.

    As such, this workflow does not produce derivatives (qsirecon_suffix is ignored).
    The map file will instead be included in the derivatives of the autotrack workflow.

    Inputs

        fibgz
            A DSI Studio fib file produced by DSI Studio reconstruction.

    Outputs

        fibgz
            The input fibgz file, unaltered
        fibgz_map
            A map.gz file corresponding to the fibgz file

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["fibgz"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fibgz", "fibgz_map", "recon_scalars"]),
        name="outputnode",
    )
    outputnode.inputs.recon_scalars = []

    omp_nthreads = config.nipype.omp_nthreads
    workflow = Workflow(name=name)

    dsi_studio_version = params.pop("dsi_studio_version", "hou")

    _AutoTrack = AutoTrack if dsi_studio_version == "hou" else ChenAutoTrack

    # Run autotrack on only one bundle. The important part is getting the map.gz
    registration_atk = pe.Node(
        _AutoTrack(num_threads=omp_nthreads, track_id="Association_ArcuateFasciculusL"),
        name="registration_atk",
        n_procs=omp_nthreads,
    )

    workflow.connect([
        (inputnode, registration_atk, [('fibgz', 'fib_file')]),
        (inputnode, outputnode, [('fibgz', 'fibgz')]),
        (registration_atk, outputnode, [('map_file', 'fibgz_map')]),
    ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)


def init_dsi_studio_autotrack_wf(
    inputs_dict,
    params={},
    qsirecon_suffix="",
    name="dsi_studio_autotrack_wf",
):
    """Run DSI Studio's AutoTrack method to produce bundles and bundle stats.

    Inputs

        fibgz
            A DSI Studio fib file produced by DSI Studio reconstruction.

    Outputs

        tck_files
            MRtrix3 format tck files for each bundle
        bundle_names
            Names that describe which bundles are present in `tck_files`

    Params:

        track_id: str
            specify the id number or the name of the bundle. The id can be found in
            /atlas/ICBM152/HCP1065.tt.gz.txt . This text file is included in DSI
            Studio package (For Mac, right-click on dsi_studio_64.app to find
            content). You can specify partial name of the bundle:

            example:
            for tracking left and right arcuate fasciculus, assign
            --track_id=0,1 or --track_id=arcuate (DSI Studio will find bundles
            with names containing "arcuate", case insensitive)

            example:
            for tracking left and right arcuate and cingulum, assign
            -track_id=0,1,2,3 or -track_id=arcuate,cingulum

        track_voxel_ratio: float
            the track-voxel ratio for the total number of streamline count. A larger
            value gives better mapping with the expense of computation time. (default: 2.0)

        tolerance: str
            the tolerance for the bundle recognition. The unit is in mm. Multiple values
            can be assigned using comma separator. A larger value may include larger track
            variation but also subject to more false results. (default: "22,26,30")

        yield_rate: float
            This rate will be used to terminate tracking early if DSI Studio finds that the
            fiber tracking is not generating results. (default: 0.00001)

        smoothing: float
            Smoothing serves like a “momentum”. For example, if smoothing is 0, the
            propagation direction is independent of the previous incoming direction.
            If the smoothing is 0.5, each moving direction remains 50% of the “momentum”,
            which is the previous propagation vector. This function makes the tracks
            appear smoother. In implementation detail, there is a weighting sum on every
            two consecutive moving directions. For smoothing value 0.2, each subsequent
            direction has 0.2 weightings contributed from the previous moving direction
            and 0.8 contributed from the income direction. To disable smoothing set
            its value to 0. Assign 1.0 to do a random selection of the value from 0% to 95%.

        otsu_threshold: float
            The ratio of otsu threshold to derive default anisotropy threshold.

        model_name: str
            The name of the model used for ODFs (default "gqi")
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["fibgz", "fibgz_map"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["tck_files", "bundle_names", "recon_scalars"]),
        name="outputnode",
    )
    outputnode.inputs.recon_scalars = []
    desc = (
        "#### DSI Studio Automatic Tractography\n\nAutomatic Tractography was run in "
        "DSI Studio (version %s) and bundle shape statistics were calculated [@autotrack]. "
        % DSI_STUDIO_VERSION
    )
    model_name = params.pop("model", "gqi")
    omp_nthreads = config.nipype.omp_nthreads
    dsi_studio_version = params.pop("dsi_studio_version", "hou")

    bundle_names = _get_dsi_studio_bundles(params.get("track_id", ""), version=dsi_studio_version)
    bundle_desc = (
        "AutoTrack attempted to reconstruct the following bundles:\n  * "
        + "\n  * ".join(bundle_names)
        + "\n\n"
    )
    LOGGER.info(bundle_desc)

    workflow = Workflow(name=name)
    workflow.__desc__ = desc + bundle_desc

    _AutoTrack = AutoTrack if dsi_studio_version == "hou" else ChenAutoTrack

    # Run autotrack!
    actual_trk = pe.Node(
        _AutoTrack(num_threads=omp_nthreads, **params), name="actual_trk", n_procs=omp_nthreads
    )  # An extra thread is needed

    # Create a single output
    aggregate_atk_results = pe.Node(
        AggregateAutoTrackResults(expected_bundles=bundle_names), name="aggregate_atk_results"
    )

    convert_to_tck = pe.MapNode(DSIStudioTrkToTck(), name="convert_to_tck", iterfield="trk_file")

    clean_bundle_names = pe.MapNode(
        niu.Function(
            input_names=["input_string"],
            output_names=["bundle"],
            function=remove_non_alphanumeric,
        ),
        name="clean_bundle_names",
        iterfield=["input_string"],
    )

    # Save tck files of the bundles into the outputs
    ds_tckfiles = pe.MapNode(
        DerivativesDataSink(
            dismiss_entities=("desc",),
            suffix="streamlines",
            model=model_name,
            extension=".tck.gz",
            compress=True,
        ),
        iterfield=["in_file", "bundle"],
        name="ds_tckfiles",
    )

    # Save the bundle csv
    ds_bundle_csv = pe.Node(
        DerivativesDataSink(
            dismiss_entities=("desc",),
            suffix="bundlestats",
            model=model_name,
            extension=".csv",
        ),
        name="ds_bundle_csv",
        run_without_submitting=True,
    )

    # Save the mapping file
    ds_mapping = pe.Node(
        DerivativesDataSink(
            dismiss_entities=("desc",),
            suffix="dwimap",
            model=model_name,
            extension="map.gz",
            compress=True,
        ),
        name="ds_mapping",
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, actual_trk, [
            ('fibgz', 'fib_file'),
            ('fibgz_map', 'map_file')]),
        (inputnode, aggregate_atk_results, [('dwi_file', 'source_file')]),
        (inputnode, convert_to_tck, [('dwi_file', 'reference_nifti')]),
        (actual_trk, ds_mapping, [
            ('map_file', 'in_file'),
            ('dsistudiotemplate', 'dsistudiotemplate')]),
        (actual_trk, aggregate_atk_results, [
            ("native_trk_files", "trk_files"),
            ("stat_files", "stat_files"),
        ]),
        (aggregate_atk_results, convert_to_tck, [("found_bundle_files", "trk_file")]),
        (aggregate_atk_results, clean_bundle_names, [("found_bundle_names", "input_string")]),
        (clean_bundle_names, ds_tckfiles, [("bundle", "bundle")]),
        (convert_to_tck, ds_tckfiles, [("tck_file", "in_file")]),
        (aggregate_atk_results, ds_bundle_csv, [("bundle_csv", "in_file")]),
        (convert_to_tck, outputnode, [("tck_file", "tck_files")]),
        (aggregate_atk_results, outputnode, [("found_bundle_names", "bundle_names")])
    ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)


def init_dsi_studio_connectivity_wf(
    inputs_dict,
    name="dsi_studio_connectivity",
    params={},
    qsirecon_suffix="",
):
    """Calculate streamline-based connectivity matrices using DSI Studio.

    DSI Studio has a deterministic tractography algorithm that can be used to
    estimate pairwise regional connectivity. It calculates multiple connectivity
    measures.

    Inputs

        fibgz
            A DSI Studio fib file produced by DSI Studio reconstruction.
        trk_file
            a DSI Studio trk.gz file

    Outputs

        matfile
            A MATLAB-format file with numerous connectivity matrices for each
            atlas.

    Params

        fiber_count
            number of streamlines to generate. Cannot also specify seed_count
        seed_count
            Number of seeds to track from. Does not guarantee a fixed number of
            streamlines and cannot be used with the fiber_count option.
        method
            0: streamline (Euler) 4: Runge Kutta
        seed_plan
            0: = traits.Enum((0, 1), argstr="--seed_plan=%d")
        initial_dir
            Seeds begin oriented as 0: the primary orientation of the ODF 1: a random orientation
            or 2: all orientations
        connectivity_type
            "pass" to count streamlines passing through a region. "end" to force
            streamlines to terminate in regions they count as connecting.
        connectivity_value
            "count", "ncount", "fa" used to quantify connection strength.
        random_seed
            Setting to True generates truly random (not-reproducible) seeding.
        fa_threshold
            If not specified, will use the DSI Studio Otsu threshold. Otherwise
            specigies the minimum qa value per fixed to be used for tracking.
        step_size
            Streamline propagation step size in millimeters.
        turning_angle
            Maximum turning angle in degrees for steamline propagation.
        smoothing
            DSI Studio smoothing factor
        min_length
            Minimum streamline length in millimeters.
        max_length
            Maximum streamline length in millimeters.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields
            + ["fibgz", "trk_file", "atlas_configs", "recon_scalars"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["matfile", "recon_scalars"]), name="outputnode"
    )
    outputnode.inputs.recon_scalars = []
    omp_nthreads = config.nipype.omp_nthreads
    plot_reports = not config.execution.skip_odf_reports

    workflow = pe.Workflow(name=name)
    calc_connectivity = pe.Node(
        DSIStudioAtlasGraph(num_threads=omp_nthreads, **params),
        name="calc_connectivity",
        n_procs=omp_nthreads,
    )

    workflow.connect([
        (inputnode, calc_connectivity, [
            ('atlas_configs', 'atlas_configs'),
            ('fibgz', 'input_fib'),
            ('trk_file', 'trk_file'),
        ]),
        (calc_connectivity, outputnode, [('connectivity_matfile', 'matfile')]),
    ])  # fmt:skip

    if plot_reports:
        plot_connectivity = pe.Node(ConnectivityReport(), name="plot_connectivity")
        ds_report_connectivity = pe.Node(
            DerivativesDataSink(
                desc="DSIStudioConnectivity",
                suffix="matrices",
                extension=".svg",
            ),
            name="ds_report_connectivity",
            run_without_submitting=True,
        )
        workflow.connect([
            (calc_connectivity, plot_connectivity, [
                ('connectivity_matfile', 'connectivity_matfile'),
            ]),
            (plot_connectivity, ds_report_connectivity, [('out_report', 'in_file')]),
        ])  # fmt:skip

    if qsirecon_suffix:
        # Save the output in the outputs directory
        ds_connectivity = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="connectivity",
            ),
            name=f"ds_{name}",
            run_without_submitting=True,
        )
        workflow.connect([
            (calc_connectivity, ds_connectivity, [('connectivity_matfile', 'in_file')]),
        ])  # fmt:skip

    return clean_datasinks(workflow, qsirecon_suffix)


def init_dsi_studio_export_wf(
    inputs_dict,
    name="dsi_studio_export",
    params={},
    qsirecon_suffix="",
):
    """Export scalar maps from a DSI Studio fib file into NIfTI files with correct headers.

    This workflow exports gfa, fa0, fa1, fa2 and iso.

    Inputs

        fibgz
            A DSI Studio fib file

    Outputs

        gfa
            NIfTI file containing generalized fractional anisotropy (GFA).
        fa0
            Quantitative Anisotropy for the largest fixel in each voxel.
        fa1
            Quantitative Anisotropy for the second-largest fixel in each voxel.
        fa2
            Quantitative Anisotropy for the third-largest fixel in each voxel.
        iso
            Isotropic component of the ODF in each voxel.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["fibgz"]), name="inputnode"
    )
    plot_reports = params.pop("plot_reports", True)  # noqa: F841
    scalar_names = [
        "qa",
        "dti_fa",
        "txx",
        "txy",
        "txz",
        "tyy",
        "tyz",
        "tzz",
        "rd1",
        "rd2",
        "ha",
        "md",
        "ad",
        "rd",
        "gfa",
        "iso",
    ]
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[name + "_file" for name in scalar_names] + ["recon_scalars"]
        ),
        name="outputnode",
    )
    workflow = pe.Workflow(name=name)
    export = pe.Node(DSIStudioExport(to_export=",".join(scalar_names)), name="export")
    recon_scalars = pe.Node(
        DSIStudioReconScalars(qsirecon_suffix=qsirecon_suffix), name="recon_scalars", n_procs=1
    )
    fixhdr_nodes = {}
    for scalar_name in scalar_names:
        output_name = scalar_name + "_file"
        fixhdr_nodes[scalar_name] = pe.Node(FixDSIStudioExportHeader(), name="fix_" + scalar_name)
        connections = [
            (export, fixhdr_nodes[scalar_name], [(output_name, "dsi_studio_nifti")]),
            (inputnode, fixhdr_nodes[scalar_name], [("dwi_file", "correct_header_nifti")]),
            (fixhdr_nodes[scalar_name], outputnode, [("out_file", output_name)]),
            (fixhdr_nodes[scalar_name], recon_scalars, [("out_file", output_name)]),
        ]
        workflow.connect(connections)  # fmt:skip

    if qsirecon_suffix:
        scalar_output_wf = init_scalar_output_wf()
        workflow.connect([
            (inputnode, scalar_output_wf, [("dwi_file", "inputnode.source_file")]),
            (recon_scalars, scalar_output_wf, [("scalar_info", "inputnode.scalar_configs")]),
            (scalar_output_wf, outputnode, [("outputnode.scalar_configs", "recon_scalars")]),
        ])  # fmt:skip

        plot_scalars = pe.Node(
            ScalarReport(),
            name="plot_scalars",
            n_procs=1,
        )
        workflow.connect([
            (inputnode, plot_scalars, [
                ("acpc_preproc", "underlay"),
                ("acpc_seg", "dseg"),
                ("dwi_mask", "mask_file"),
            ]),
            (recon_scalars, plot_scalars, [("scalar_info", "scalar_metadata")]),
            (scalar_output_wf, plot_scalars, [("outputnode.scalar_files", "scalar_maps")]),
        ])  # fmt:skip

        ds_report_scalars = pe.Node(
            DerivativesDataSink(
                datatype="figures",
                desc="scalars",
                suffix="dwimap",
                dismiss_entities=["dsistudiotemplate"],
            ),
            name="ds_report_scalars",
            run_without_submitting=True,
        )
        workflow.connect([(plot_scalars, ds_report_scalars, [("out_report", "in_file")])])
    else:
        # If not writing out scalar files, pass the working directory scalar configs
        workflow.connect([(recon_scalars, outputnode, [("scalar_info", "recon_scalars")])])

    workflow.connect([(inputnode, export, [("fibgz", "input_file")])])

    return clean_datasinks(workflow, qsirecon_suffix)
