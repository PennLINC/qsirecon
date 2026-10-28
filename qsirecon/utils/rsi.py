"""Restriction Spectrum Imaging (RSI) fitting.

This is a port of ``RSIproc`` 1.0.8 from the ABCD Study
(https://github.com/ABCD-STUDY/RSI, Richard Watts),
Copyright (c) 2024 ABCD Data Analytics and Informatics Resource Center,
distributed under the MIT License (see ``qsirecon/data/rsi/LICENSE``).

The upstream script hard-codes its model settings and loops over voxels in Python.
This port takes the settings as arguments and fits all voxels at once.
The unmodified upstream script is kept in ``qsirecon/data/rsi`` as the reference
implementation, and ``qsirecon/tests/test_rsi.py`` checks that this port reproduces it.
"""

import numpy as np

try:
    from scipy.special import sph_harm_y
except ImportError:  # scipy < 1.15
    from scipy.special import sph_harm

    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)


RSI_VERSION = '1.0.8'
RSI_COMMIT = 'f775121a4ca9a0d8e9e355c60ce85b4fd7d879ac'
RSI_URL = f'https://github.com/ABCD-STUDY/RSI/tree/{RSI_COMMIT}'
RSI_SCRIPT = 'rsi/RSIproc_{}.py'.format(RSI_VERSION.replace('.', '_'))

# The model settings hard-coded in RSIproc, which are used as defaults.
# Compartments are restricted (s1), hindered (s2) and free water (s3).
RSI_COMPARTMENTS = ['restricted', 'hindered', 'free']
RSI_LAMBDA = 0.1
RSI_ADC_LONG = [1.0e-3, 1.0e-3, 3.0e-3]
RSI_ADC_TRANS = [1.0e-10, 0.9e-3, 3.0e-3]
RSI_SH_ORDER = [4, 4, 0]

# RSIproc samples the response functions on an icosahedron subdivided this many times
ICOSAHEDRON_ORDER = 3


def make_icosahedron(order=ICOSAHEDRON_ORDER):
    """Return the vertices of RSIproc's subdivided icosahedron.

    Each subdivision splits every face into four, adding the normalized midpoint of each edge.
    This reproduces the vertex set of RSIproc's ``icosahedron.make_icosahedron``.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    vertices = [np.array(v, dtype=float) / np.linalg.norm(v) for v in vertices]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt:skip

    for _ in range(order):
        # Index of the vertex at the midpoint of each edge, so shared edges add one vertex
        midpoints = {}
        new_faces = []
        for v0, v1, v2 in faces:
            face_midpoints = []
            for i, j in ((v0, v1), (v1, v2), (v2, v0)):
                edge = (min(i, j), max(i, j))
                if edge not in midpoints:
                    midpoint = (vertices[i] + vertices[j]) / 2
                    vertices.append(midpoint / np.linalg.norm(midpoint))
                    midpoints[edge] = len(vertices) - 1
                face_midpoints.append(midpoints[edge])
            v3, v4, v5 = face_midpoints
            new_faces += [[v0, v3, v5], [v3, v1, v4], [v4, v2, v5], [v3, v4, v5]]
        faces = new_faces

    return np.array(vertices)


def real_sh_basis(max_degree, directions):
    """Evaluate RSIproc's real, even-degree spherical harmonic basis.

    Parameters
    ----------
    max_degree : int
        Maximum (even) spherical harmonic degree.
    directions : (M, 3) array
        Unit vectors at which to evaluate the basis.

    Returns
    -------
    (M, (max_degree + 1) * (max_degree + 2) // 2) array
        Basis functions ordered by degree, then by order from -degree to degree.
    """
    polar = np.arccos(directions[:, 2])
    azimuth = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi)
    n_coefs = (max_degree + 1) * (max_degree + 2) // 2
    basis = np.zeros((directions.shape[0], n_coefs))
    for degree in range(0, max_degree + 1, 2):
        for order in range(-degree, degree + 1):
            col = (degree * degree + degree) // 2 + order
            ylm = sph_harm_y(degree, order, polar, azimuth)
            if order < 0:
                basis[:, col] = np.sqrt(2.0) * ylm.real
            elif order == 0:
                basis[:, col] = ylm.real
            else:
                basis[:, col] = np.sqrt(2.0) * ylm.imag
    return basis


def _n_sh_coefs(sh_degree):
    return (sh_degree + 1) * (sh_degree + 2) // 2


def rsi_design_matrix(
    bvals, bvecs, parallel_diffusivities, perpendicular_diffusivities, sh_degrees
):
    """Build the RSI design matrix that maps SH coefficients to normalized signal.

    Each compartment's signal is an axially symmetric tensor response, integrated over a
    fiber orientation distribution represented with spherical harmonics up to its degree.
    """
    vertices = make_icosahedron()
    basis = real_sh_basis(max(sh_degrees), vertices)
    columns = []
    for d_par, d_perp, sh_degree in zip(
        parallel_diffusivities, perpendicular_diffusivities, sh_degrees
    ):
        cos_angle = bvecs @ vertices.T
        response = np.exp(-bvals[:, None] * ((d_par - d_perp) * cos_angle**2 + d_perp))
        columns.append(response @ np.linalg.pinv(basis[:, : _n_sh_coefs(sh_degree)]).T)
    return np.hstack(columns) * basis[0, 0]


def fit_rsi(
    data,
    mask,
    bvals,
    bvecs,
    parallel_diffusivities=RSI_ADC_LONG,
    perpendicular_diffusivities=RSI_ADC_TRANS,
    sh_degrees=RSI_SH_ORDER,
    regularization_lambda=RSI_LAMBDA,
    b0_threshold=50.0,
):
    """Fit RSI to a DWI series.

    Parameters
    ----------
    data : (X, Y, Z, N) array
        Diffusion-weighted images.
    mask : (X, Y, Z) array
        Voxels to fit.
    bvals : (N,) array
        b-values in s/mm^2. Values below ``b0_threshold`` are treated as b=0.
    bvecs : (N, 3) array
        Gradient directions.
    parallel_diffusivities, perpendicular_diffusivities : sequence of float
        Diffusivity of each compartment parallel and perpendicular to the fiber, in mm^2/s.
    sh_degrees : sequence of int
        Maximum (even) spherical harmonic degree of each compartment's orientation distribution.
        A degree of 0 makes the compartment isotropic.
    regularization_lambda : float
        Weight of the Tikhonov regularization, relative to the mean diagonal of A^T A.
    b0_threshold : float
        b-values below this are treated as b=0.

    Returns
    -------
    dict
        ``iso``, ``dir`` and ``total`` are (n_compartments, X, Y, Z) arrays holding the norms of
        each compartment's degree 0, degree > 0 and all SH coefficients.
        ``dir`` is 0 for isotropic compartments.
        ``l2norm`` is the (X, Y, Z) norm of all coefficients.
        As in RSIproc, coefficients are not normalized by ``l2norm``
        and ``iso`` is clipped to [-1, 2].
    """
    bvals = np.asarray(bvals, dtype=float).copy()
    b0_mask = bvals < b0_threshold
    if not b0_mask.any():
        raise ValueError(
            f'RSI requires at least one b=0 image, but no b-values are below {b0_threshold}.'
        )
    bvals[b0_mask] = 0

    design = rsi_design_matrix(
        bvals, bvecs, parallel_diffusivities, perpendicular_diffusivities, sh_degrees
    )
    ata = design.T @ design
    scale = np.mean(np.diag(ata))
    weights = np.linalg.inv(ata + regularization_lambda * scale * np.eye(ata.shape[0])) @ design.T

    b0_mean = data[..., b0_mask].astype(float).mean(axis=-1)
    fit_mask = (mask > 0) & (b0_mean > 0)
    signal = data[fit_mask].astype(float) / b0_mean[fit_mask, None]
    coefs = signal @ weights.T

    spatial_shape = data.shape[:3]
    n_compartments = len(sh_degrees)
    results = {
        'iso': np.zeros((n_compartments,) + spatial_shape),
        'dir': np.zeros((n_compartments,) + spatial_shape),
        'total': np.zeros((n_compartments,) + spatial_shape),
        'l2norm': np.zeros(spatial_shape),
    }
    start = 0
    for i_comp, sh_degree in enumerate(sh_degrees):
        stop = start + _n_sh_coefs(sh_degree)
        results['iso'][i_comp][fit_mask] = coefs[:, start]
        if sh_degree > 0:
            results['dir'][i_comp][fit_mask] = np.linalg.norm(coefs[:, start + 1 : stop], axis=1)
        results['total'][i_comp][fit_mask] = np.linalg.norm(coefs[:, start:stop], axis=1)
        start = stop
    results['l2norm'][fit_mask] = np.linalg.norm(coefs, axis=1)

    # RSIproc clips impossible values of the isotropic components, which should be 0-1
    np.clip(results['iso'], -1.0, 2.0, out=results['iso'])
    return results
