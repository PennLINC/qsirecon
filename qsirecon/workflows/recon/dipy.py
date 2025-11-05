"""
Dipy Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dipy_brainsuite_shore_recon_wf
.. autofunction:: init_dipy_mapmri_recon_wf
.. autofunction:: init_dipy_dki_recon_wf

"""

import logging

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.bids import DerivativesDataSink
from ...interfaces.dipy import (
    BrainSuiteShoreReconstruction,
    KurtosisReconstruction,
    KurtosisReconstructionMicrostructure,
    MAPMRIReconstruction,
)
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import (
    ReconScalars,
    brainsuite_3dshore_scalars,
    dipy_dki_scalars,
    dipy_mapmri_scalars,
)
from ...interfaces.reports import CLIReconPeaksReport, ScalarReport
from ...utils.bids import clean_datasinks
from .utils import init_scalar_output_wf

LOGGER = logging.getLogger("nipype.interface")


def external_format_datasinks(qsirecon_suffix, params, wf):
    """Add datasinks for Dipy Reconstructions in other formats."""
    outputnode = wf.get_node("outputnode")
    if params["write_fibgz"]:
        ds_fibgz = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwimap",
                extension=".fib.gz",
                compress=True,
            ),
            name=f"ds_{qsirecon_suffix}_fibgz",
            run_without_submitting=True,
        )
        wf.connect(outputnode, 'fibgz',
                   ds_fibgz, 'in_file')  # fmt:skip

    if params["write_mif"]:
        ds_mif = pe.Node(
            DerivativesDataSink(
                dismiss_entities=("desc",),
                suffix="dwimap",
                extension=".mif",
                compress=False,
            ),
            name=f"ds_{qsirecon_suffix}_mif",
            run_without_submitting=True,
        )
        wf.connect(outputnode, 'fod_sh_mif',
                   ds_mif, 'in_file')  # fmt:skip


def init_dipy_brainsuite_shore_recon_wf(
    inputs_dict,
    name="dipy_3dshore_recon",
    qsirecon_suffix="",
    params={},
):
    """Reconstruct EAPs, ODFs, using 3dSHORE (brainsuite-style basis set).

    Inputs

        *qsirecon outputs*

    Outputs

        shore_coeffs
            3dSHORE coefficients
        rtop
            Voxelwise Return-to-origin probability.
        rtap
            Voxelwise Return-to-axis probability.
        rtpp
            Voxelwise Return-to-plane probability.


    Params

        write_fibgz: bool
            True writes out a DSI Studio fib file
        write_mif: bool
            True writes out a MRTrix mif file with sh coefficients
        convert_to_multishell: str
            either "HCP", "ABCD", "lifespan" will resample the data with this scheme
        radial_order: int
            Radial order for spherical harmonics (even)
        zeta: float
            Zeta parameter for basis set.
        tau:float
            Diffusion parameter (default= 4 * np.pi**2)
        regularization
            "L2" or "L1". Default is "L2"
        lambdaN
            LambdaN parameter for L2 regularization. (default=1e-8)
        lambdaL
            LambdaL parameter for L2 regularization. (default=1e-8)
        regularization_weighting: int or "CV"
            L1 regualrization weighting. Default "CV" (use cross-validation).
            Can specify a static value to use in all voxels.
        l1_positive_constraint: bool
            Use positivity constraint.
        l1_maxiter
            Maximum number of iterations for L1 optization. (Default=1000)
        l1_alpha
            Alpha parameter for L1 optimization. (default=1.0)
        pos_grid: int
            Grid points for estimating EAP(default=11)
        pos_radius
            Radius for EAP estimation (default=20e-03)

    """
    omp_nthreads = config.nipype.omp_nthreads
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "shore_coeffs_image",
                "rtop_image",
                "alpha_image",
                "r2_image",
                "cnr_image",
                "regularization_image",
                "fibgz",
                "fod_sh_mif",
                "dwi_file",
                "bval_file",
                "bvec_file",
                "b_file",
                "recon_scalars",
            ]
        ),
        name="outputnode",
    )
    plot_reports = not config.execution.skip_odf_reports
    workflow = Workflow(name=name)
    desc = "#### Dipy Reconstruction\n\n"

    # Do we have deltas?
    deltas = (params.get("big_delta", None), params.get("small_delta", None))
    approximate_deltas = None in deltas
    dwi_metadata = inputs_dict.get("dwi_metadata", {})
    if approximate_deltas:
        deltas = (
            dwi_metadata.get("LargeDelta", None),
            dwi_metadata.get("SmallDelta", None),
        )
        approximate_deltas = None in deltas

    # Set deltas if we have them. Prevent only one from being defined
    if approximate_deltas:
        LOGGER.warning('Both "big_delta" and "small_delta" are required for precise 3dSHORE')
    else:
        params["big_delta"], params["small_delta"] = deltas

    recon_shore = pe.Node(BrainSuiteShoreReconstruction(**params), name="recon_shore")
    recon_scalars = pe.Node(
        ReconScalars(scalar_config=brainsuite_3dshore_scalars, qsirecon_suffix="name"),
        name="recon_scalars",
        run_without_submitting=True,
    )
    doing_extrapolation = params.get("extrapolate_scheme") in ("HCP", "ABCD", "DSIQ5")

    workflow.connect([
        (inputnode, recon_shore, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file')]),
        (recon_shore, outputnode, [
            ('shore_coeffs_image', 'shore_coeffs_image'),
            ('rtop_image', 'rtop_image'),
            ('alpha_image', 'alpha_image'),
            ('r2_image', 'r2_image'),
            ('cnr_image', 'cnr_image'),
            ('regularization_image', 'regularization_image'),
            ('fibgz', 'fibgz'),
            ('fod_sh_mif', 'fod_sh_mif'),
            ('extrapolated_dwi', 'dwi_file'),
            ('extrapolated_bvals', 'bval_file'),
            ('extrapolated_bvecs', 'bvec_file'),
            ('extrapolated_b', 'b_file')]),
        (recon_shore, recon_scalars, [
            ('rtop_image', 'rtop_file'),
            ('alpha_image', 'alpha_image'),
            ('r2_image', 'r2_image'),
            ('cnr_image', 'cnr_image'),
            ('regularization_image', 'regularization_image')]),
        (recon_scalars, outputnode, [("scalar_info", "recon_scalars")])
    ])  # fmt:skip

    if plot_reports:
        plot_peaks = pe.Node(
            CLIReconPeaksReport(),
            name="plot_peaks",
            n_procs=omp_nthreads,
        )
        ds_report_peaks = pe.Node(
            DerivativesDataSink(
                desc="3dSHOREODF",
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
            (recon_shore, plot_peaks, [
                ('odf_directions', 'directions_file'),
                ('odf_amplitudes', 'odf_file'),
            ]),
            (plot_peaks, ds_report_peaks, [('peak_report', 'in_file')]),
        ])  # fmt:skip

    # Plot targeted regions
    if inputs_dict["has_qsiprep_t1w_transforms"] and plot_reports:
        ds_report_odfs = pe.Node(
            DerivativesDataSink(
                desc="3dSHOREODF",
                suffix="odfs",
                extension=".png",
            ),
            name="ds_report_odfs",
            run_without_submitting=True,
        )
        workflow.connect([(plot_peaks, ds_report_odfs, [("odf_report", "in_file")])])

    if qsirecon_suffix:
        external_format_datasinks(qsirecon_suffix, params, workflow)

        ds_rtop = pe.Node(
            DerivativesDataSink(
                desc="rtop",
                extension=".nii.gz",
                compress=True,
            ),
            name="ds_bsshore_rtop",
            run_without_submitting=True,
        )
        workflow.connect(outputnode, 'rtop_image',
                         ds_rtop, 'in_file')  # fmt:skip

        ds_coeff = pe.Node(
            DerivativesDataSink(
                desc="SHOREcoeff",
                extension=".nii.gz",
                compress=True,
            ),
            name="ds_bsshore_coeff",
            run_without_submitting=True,
        )
        workflow.connect(outputnode, 'shore_coeffs_image',
                         ds_coeff, 'in_file')  # fmt:skip

        ds_alpha = pe.Node(
            DerivativesDataSink(
                desc="L1alpha",
                extension=".nii.gz",
                compress=True,
            ),
            name="ds_bsshore_alpha",
            run_without_submitting=True,
        )
        workflow.connect(outputnode, 'alpha_image',
                         ds_alpha, 'in_file')  # fmt:skip

        ds_r2 = pe.Node(
            DerivativesDataSink(
                desc="r2",
                extension=".nii.gz",
                compress=True,
            ),
            name="ds_bsshore_r2",
            run_without_submitting=True,
        )
        workflow.connect(outputnode, 'r2_image',
                         ds_r2, 'in_file')  # fmt:skip

        ds_cnr = pe.Node(
            DerivativesDataSink(
                desc="CNR",
                extension=".nii.gz",
                compress=True,
            ),
            name="ds_bsshore_cnr",
            run_without_submitting=True,
        )
        workflow.connect(outputnode, 'cnr_image',
                         ds_cnr, 'in_file')  # fmt:skip

        ds_regl = pe.Node(
            DerivativesDataSink(
                desc="regularization",
                extension=".nii.gz",
                compress=True,
            ),
            name="ds_bsshore_regl",
            run_without_submitting=True,
        )
        workflow.connect(outputnode, 'regularization_image',
                         ds_regl, 'in_file')  # fmt:skip
        if doing_extrapolation:
            ds_extrap_dwi = pe.Node(
                DerivativesDataSink(
                    desc="extrapolated",
                    extension=".nii.gz",
                    compress=True,
                ),
                name="ds_extrap_dwi",
                run_without_submitting=True,
            )
            workflow.connect(outputnode, 'dwi_file',
                             ds_extrap_dwi, 'in_file')  # fmt:skip
            ds_extrap_bval = pe.Node(
                DerivativesDataSink(
                    desc="extrapolated",
                    extension=".bval",
                ),
                name="ds_extrap_bval",
                run_without_submitting=True,
            )
            workflow.connect(outputnode, 'bval_file',
                             ds_extrap_bval, 'in_file')  # fmt:skip
            ds_extrap_bvec = pe.Node(
                DerivativesDataSink(
                    desc="extrapolated",
                    extension=".bvec",
                ),
                name="ds_extrap_bvec",
                run_without_submitting=True,
            )
            workflow.connect(outputnode, 'bvec_file',
                             ds_extrap_bvec, 'in_file')  # fmt:skip
            ds_extrap_b = pe.Node(
                DerivativesDataSink(
                    desc="extrapolated",
                    extension=".b",
                ),
                name="ds_extrap_b",
                run_without_submitting=True,
            )
            workflow.connect(outputnode, 'b_file',
                             ds_extrap_b, 'in_file')  # fmt:skip
    workflow.__desc__ = desc

    return clean_datasinks(workflow, qsirecon_suffix)


def init_dipy_mapmri_recon_wf(
    inputs_dict,
    name="dipy_mapmri_recon",
    qsirecon_suffix="",
    params={},
):
    """Reconstruct EAPs, ODFs, using 3dSHORE (brainsuite-style basis set).

    Inputs

        *qsirecon outputs*

    Outputs

        shore_coeffs
            3dSHORE coefficients
        rtop
            Voxelwise Return-to-origin probability.
        rtap
            Voxelwise Return-to-axis probability.
        rtpp
            Voxelwise Return-to-plane probability.
        msd
            Voxelwise MSD
        qiv
            q-space inverse variance
        lapnorm
            Voxelwise norm of the Laplacian

    Params

        write_fibgz: bool
            True writes out a DSI Studio fib file
        write_mif: bool
            True writes out a MRTrix mif file with sh coefficients
        radial_order: int
            An even integer that represent the order of the basis
        laplacian_regularization: bool
            Regularize using the Laplacian of the MAP-MRI basis.
        laplacian_weighting: str or scalar
            The string 'GCV' makes it use generalized cross-validation to find
            the regularization weight. A scalar sets the regularization
            weight to that value and an array will make it selected the
            optimal weight from the values in the array.
        positivity_constraint: bool
            Constrain the propagator to be positive.
        pos_grid: int
            Grid points for estimating EAP(default=15)
        pos_radius
            Radius for EAP estimation (default=20e-03) or "adaptive"
        anisotropic_scaling : bool,
            If True, uses the standard anisotropic MAP-MRI basis. If False,
            uses the isotropic MAP-MRI basis (equal to 3D-SHORE).
        eigenvalue_threshold : float,
            Sets the minimum of the tensor eigenvalues in order to avoid
            stability problem.
        bval_threshold : float,
            Sets the b-value threshold to be used in the scale factor
            estimation. In order for the estimated non-Gaussianity to have
            meaning this value should set to a lower value (b<2000 s/mm^2)
            such that the scale factors are estimated on signal points that
            reasonably represent the spins at Gaussian diffusion.
        dti_scale_estimation : bool,
            Whether or not DTI fitting is used to estimate the isotropic scale
            factor for isotropic MAP-MRI.
            When set to False the algorithm presets the isotropic tissue
            diffusivity to static_diffusivity. This vastly increases fitting
            speed but at the cost of slightly reduced fitting quality. Can
            still be used in combination with regularization and constraints.
        static_diffusivity : float,
            the tissue diffusivity that is used when dti_scale_estimation is
            set to False. The default is that of typical white matter
            D=0.7e-3 _[5].
        cvxpy_solver : str, optional
            cvxpy solver name. Optionally optimize the positivity constraint
            with a particular cvxpy solver. See http://www.cvxpy.org/ for
            details.
            Default: None (cvxpy chooses its own solver)
    """

    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "mapmri_coeffs",
                "rtop",
                "rtap",
                "rtpp",
                "fibgz",
                "fod_sh_mif",
                "parng",
                "perng",
                "ng",
                "qiv",
                "lapnorm",
                "msd",
                "recon_scalars",
            ]
        ),
        name="outputnode",
    )

    workflow = Workflow(name=name)
    desc = "#### Dipy Reconstruction\n\n"

    # Do we have deltas?
    deltas = (params.get("big_delta", None), params.get("small_delta", None))
    approximate_deltas = None in deltas
    dwi_metadata = inputs_dict.get("dwi_metadata", {})
    if approximate_deltas:
        deltas = (
            dwi_metadata.get("LargeDelta", None),
            dwi_metadata.get("SmallDelta", None),
        )
        approximate_deltas = None in deltas

    # Set deltas if we have them. Prevent only one from being defined
    if approximate_deltas:
        LOGGER.warning('Both "big_delta" and "small_delta" are required for precise MAPMRI')
    else:
        params["big_delta"], params["small_delta"] = deltas

    plot_reports = not config.execution.skip_odf_reports
    omp_nthreads = config.nipype.omp_nthreads
    recon_map = pe.Node(MAPMRIReconstruction(**params), name="recon_map")
    recon_scalars = pe.Node(
        ReconScalars(scalar_config=dipy_mapmri_scalars, dismiss_entities=["desc"], qsirecon_suffix=name),
        name="recon_scalars",
        run_without_submitting=True,
    )
    workflow.connect([
        (inputnode, recon_map, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file')]),
        (recon_map, outputnode, [
            ('mapmri_coeffs', 'mapmri_coeffs'),
            ('rtop', 'rtop'),
            ('rtap', 'rtap'),
            ('rtpp', 'rtpp'),
            ('parng', 'parng'),
            ('perng', 'perng'),
            ('msd', 'msd'),
            ('ng', 'ng'),
            ('qiv', 'qiv'),
            ('lapnorm', 'lapnorm'),
            ('fibgz', 'fibgz'),
            ('fod_sh_mif', 'fod_sh_mif')]),
        (recon_map, recon_scalars, [
            ('rtop', 'rtop_file'),
            ('rtap', 'rtap_file'),
            ('rtpp', 'rtpp_file'),
            ('ng', 'ng_file'),
            ('parng', 'ngpar_file'),
            ('perng', 'ngperp_file'),
            ('msd', 'msd_file'),
            ('qiv', 'qiv_file'),
            ('lapnorm', 'lapnorm_file'),
            ('mapmri_coeffs', 'mapcoeffs_file'),
        ]),
    ])  # fmt:skip

    if plot_reports:
        plot_peaks = pe.Node(
            CLIReconPeaksReport(),
            name="plot_peaks",
            n_procs=omp_nthreads,
        )
        ds_report_peaks = pe.Node(
            DerivativesDataSink(
                desc="MAPLMRIODF",
                suffix="peaks",
                extension=".png",
            ),
            name="ds_report_peaks",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, plot_peaks, [
                ('dwi_mask', 'mask_file'),
                ('dwi_ref', 'background_image'),
                ('odf_rois', 'odf_rois')]),
            (recon_map, plot_peaks, [
                ('odf_directions', 'directions_file'),
                ('odf_amplitudes', 'odf_file')]),
            (plot_peaks, ds_report_peaks, [('peak_report', 'in_file')]),
        ])  # fmt:skip

    # Plot targeted regions
    if inputs_dict["has_qsiprep_t1w_transforms"] and plot_reports:
        ds_report_odfs = pe.Node(
            DerivativesDataSink(
                desc="MAPLMRIODF",
                suffix="odfs",
                extension=".png",
            ),
            name="ds_report_odfs",
            run_without_submitting=True,
        )
        workflow.connect([(plot_peaks, ds_report_odfs, [("odf_report", "in_file")])])

    if qsirecon_suffix:
        external_format_datasinks(qsirecon_suffix, params, workflow)

        scalar_output_wf = init_scalar_output_wf()
        workflow.connect([
            (inputnode, scalar_output_wf, [("dwi_file", "inputnode.source_file")]),
            (recon_scalars, scalar_output_wf, [("scalar_info", "inputnode.scalar_configs")]),
        ])  # fmt:skip

        plot_scalars = pe.Node(
            ScalarReport(),
            name="plot_scalars",
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, plot_scalars, [
                ("acpc_preproc", "underlay"),
                ("acpc_seg", "dseg"),
                ("dwi_mask", "mask_file"),
            ]),
            (recon_scalars, plot_scalars, [("scalar_info", "scalar_metadata")]),
            (scalar_output_wf, plot_scalars, [("outputnode.scalar_files", "scalar_maps")]),
            (scalar_output_wf, outputnode, [("outputnode.scalar_configs", "recon_scalars")]),
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

    workflow.__desc__ = desc

    return clean_datasinks(workflow, qsirecon_suffix)


def init_dipy_dki_recon_wf(inputs_dict, name="dipy_dki_recon", qsirecon_suffix="", params={}):
    """Fit DKI

    Inputs

        *qsirecon outputs*

    Outputs

        tensor
        fa
        md
        rd
        ad
        color_fa
        kfa
        mk
        ak
        rk
        mkt
        awf
            Only if wmti is True
        rde
            Only if wmti is True

    Params

        write_fibgz: bool
            True writes out a DSI Studio fib file
        write_mif: bool
            True writes out a MRTrix mif file with sh coefficients
        radial_order: int
            An even integer that represent the order of the basis

    """

    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "tensor",
                "colorFA",
                "ad",
                "ak",
                "fa",
                "kfa",
                "linearity",
                "md",
                "mk",
                "mkt",
                "planarity",
                "rd",
                "rk",
                "sphericity",
                # Only if wmti is True
                "dkimicro_ad",
                "dkimicro_ade",
                "dkimicro_awf",
                "dkimicro_axonald",
                "dkimicro_kfa",
                "dkimicro_md",
                "dkimicro_rd",
                "dkimicro_rde",
                "dkimicro_tortuosity",
                "dkimicro_trace",
                # Aggregated scalars
                "recon_scalars",
            ]
        ),
        name="outputnode",
    )
    recon_scalars = pe.Node(
        ReconScalars(scalar_config=dipy_dki_scalars, qsirecon_suffix=qsirecon_suffix),
        run_without_submitting=True,
        name="recon_scalars",
    )
    workflow = Workflow(name=name)
    desc = "#### Dipy Reconstruction\n\n"

    plot_reports = not config.execution.skip_odf_reports
    micro_metrics = params.pop("wmti", False)

    recon_dki = pe.Node(KurtosisReconstruction(**params), name="recon_dki")

    workflow.connect([
        (inputnode, recon_dki, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file'),
        ]),
        (recon_dki, outputnode, [
            ('tensor', 'tensor'),
            ('fibgz', 'fibgz'),
            ('ad', 'ad'),
            ('ak', 'ak'),
            ('colorFA', 'colorFA'),
            ('fa', 'fa'),
            ('kfa', 'kfa'),
            ('linearity', 'linearity'),
            ('md', 'md'),
            ('mk', 'mk'),
            ('mkt', 'mkt'),
            ('planarity', 'planarity'),
            ('rd', 'rd'),
            ('rk', 'rk'),
            ('sphericity', 'sphericity'),
        ]),
        (recon_dki, recon_scalars, [
            ('ad', 'dki_ad'),
            ('ak', 'dki_ak'),
            ('fa', 'dki_fa'),
            ('kfa', 'dki_kfa'),
            ('linearity', 'dki_linearity'),
            ('md', 'dki_md'),
            ('mk', 'dki_mk'),
            ('mkt', 'dki_mkt'),
            ('planarity', 'dki_planarity'),
            ('rd', 'dki_rd'),
            ('rk', 'dki_rk'),
            ('sphericity', 'dki_sphericity'),
        ]),
    ])  # fmt:skip

    if micro_metrics:
        recon_dkimicro = pe.Node(
            KurtosisReconstructionMicrostructure(**params),
            name="recon_dkimicro",
        )
        # Only produce microstructural metrics if wmti is True
        workflow.connect([
            (inputnode, recon_dkimicro, [
                ('dwi_file', 'dwi_file'),
                ('bval_file', 'bval_file'),
                ('bvec_file', 'bvec_file'),
                ('dwi_mask', 'mask_file'),
            ]),
            (recon_dkimicro, outputnode, [
                ('ad', 'dkimicro_ad'),
                ('ade', 'dkimicro_ade'),
                ('awf', 'dkimicro_awf'),
                ('axonald', 'dkimicro_axonald'),
                ('kfa', 'dkimicro_kfa'),
                ('md', 'dkimicro_md'),
                ('rd', 'dkimicro_rd'),
                ('rde', 'dkimicro_rde'),
                ('tortuosity', 'dkimicro_tortuosity'),
                ('trace', 'dkimicro_trace'),
            ]),
            (recon_dkimicro, recon_scalars, [
                ('ad', 'dkimicro_ad'),
                ('ade', 'dkimicro_ade'),
                ('awf', 'dkimicro_awf'),
                ('axonald', 'dkimicro_axonald'),
                ('kfa', 'dkimicro_kfa'),
                ('md', 'dkimicro_md'),
                ('rd', 'dkimicro_rd'),
                ('rde', 'dkimicro_rde'),
                ('tortuosity', 'dkimicro_tortuosity'),
                ('trace', 'dkimicro_trace'),
            ]),
        ])  # fmt:skip

    if plot_reports and False:
        plot_peaks = pe.Node(
            CLIReconPeaksReport(peaks_only=True),
            name="plot_peaks",
            n_procs=config.nipype.omp_nthreads,
        )
        ds_report_peaks = pe.Node(
            DerivativesDataSink(
                desc="DKI",
                suffix="peaks",
                extension=".png",
            ),
            name="ds_report_peaks",
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, plot_peaks, [
                ("dwi_ref", "background_image"),
                ("odf_rois", "odf_rois"),
                ("dwi_mask", "mask_file"),
            ]),
            (recon_dki, plot_peaks, [
                ("odf_directions", "directions_file"),
                ("odf_amplitudes", "odf_file"),
            ]),
            (plot_peaks, ds_report_peaks, [("peak_report", "in_file")]),
        ])  # fmt:skip

    if qsirecon_suffix:
        external_format_datasinks(qsirecon_suffix, params, workflow)

        scalar_output_wf = init_scalar_output_wf()
        workflow.connect([
            (inputnode, scalar_output_wf, [("dwi_file", "inputnode.source_file")]),
            (recon_scalars, scalar_output_wf, [("scalar_info", "inputnode.scalar_configs")]),
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
            (scalar_output_wf, outputnode, [("outputnode.scalar_configs", "recon_scalars")]),
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

    workflow.__desc__ = desc

    return clean_datasinks(workflow, qsirecon_suffix)
