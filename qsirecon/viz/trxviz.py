"""Render ODF and peak report figures with the TRXViz headless renderer.

`TRXViz <https://github.com/tee-ar-ex/TRXViz>`_ ships ``trxviz-cli``, which renders
TRXViz workflow projects to PNG through wgpu. It needs neither an X server nor Xvfb.
It does need a Vulkan driver (TRXViz 0.6.0's OpenGL fallback cannot draw ODF glyphs or
fixels); without a GPU, Mesa's CPU Vulkan driver (lavapipe) works.

The ODFs are passed to TRXViz as an ODX directory (the format is described in
https://github.com/PennLINC/odx-rs/blob/main/SPECIFICATION.md). Only the voxels on the
slices that appear in the figures are written, so the ODX stays small.

ODF sampling directions and peak directions are treated as vectors along the voxel
axes, as the FURY-based plots that this module replaces did. They are rotated into
RAS+ world space with the image affine before they are written.
"""

import json
import os
import os.path as op
import shutil
import subprocess

import nibabel as nb
import numpy as np
from dipy.core.sphere import HemiSphere
from dipy.direction import peak_directions
from nipype import logging
from PIL import Image

from .utils import slices_from_bbox

LOGGER = logging.getLogger('nipype.interface')

TRXVIZ_CLI = 'trxviz-cli'

# Radius of the largest ODF lobe and half-length of the longest peak, as a fraction of
# the smallest voxel dimension
GLYPH_RADIUS = 0.45

# Brightness (0-1) of the brightest background voxels. The FURY-based figures drew the
# background at 60% opacity over black, so the glyphs stand out from it.
BACKGROUND_BRIGHTNESS = 0.6

TRXVIZ_ERROR = """\
trxviz-cli failed to render {png}.

TRXViz renders with wgpu and needs a working Vulkan driver. On machines without a GPU,
Mesa's CPU Vulkan driver (lavapipe, from the mesa-vulkan-drivers package) is used; the
QSIRecon container includes it. To skip these figures, rerun QSIRecon with
--skip-odf-reports.

Command: {cmd}
Return code: {returncode}
stdout:
{stdout}
stderr:
{stderr}
"""

# TRXViz world axis -> name of the slice view that is perpendicular to it
_VIEW_NAMES = ('Sagittal', 'Coronal', 'Axial')
# TRXViz orders its per-view settings as axial, coronal, sagittal
_VIEW_ORDER = ('Axial', 'Coronal', 'Sagittal')
# In-plane world axes (horizontal, vertical) of each view
_IN_PLANE_AXES = {'Axial': (0, 1), 'Coronal': (0, 2), 'Sagittal': (1, 2)}


def scale_odfs_for_display(odfs, subtract_iso=False):
    """Scale each ODF so its largest amplitude is ``GLYPH_RADIUS``.

    Parameters
    ----------
    odfs : (n_voxels, n_directions) array
    subtract_iso : bool
        Subtract each ODF's minimum first, so the glyphs show only the anisotropic part.
    """
    odfs = np.array(odfs, dtype='float32')
    if subtract_iso:
        odfs -= odfs.min(axis=1, keepdims=True)

    odfs = np.clip(odfs, 0, None)
    peaks = odfs.max(axis=1, keepdims=True)
    return np.divide(odfs * GLYPH_RADIUS, peaks, out=np.zeros_like(odfs), where=peaks > 0)


def find_trxviz_cli():
    """Return the path to ``trxviz-cli``, or raise if it is not on the ``PATH``."""
    cli = shutil.which(TRXVIZ_CLI)
    if cli is None:
        raise FileNotFoundError(
            f'{TRXVIZ_CLI} was not found on the PATH. It is needed to plot ODFs and peaks. '
            'It is included in the QSIRecon container; otherwise, download it from '
            'https://github.com/tee-ar-ex/TRXViz/releases or rerun with --skip-odf-reports.'
        )
    return cli


def fold_full_sphere(odfs, directions):
    """Fold ODFs sampled on a full, antipodally ordered sphere onto its first half.

    Some inputs carry one amplitude per vertex of a full sphere whose second half mirrors
    the first. AMICO's fib files are one: their stand-in ODFs (from
    :func:`~qsirecon.interfaces.converters.peaks_to_odfs`) put each peak on the nearest
    vertex of the first half and leave the second half empty. The larger amplitude of each
    antipodal pair is kept, so both that layout and symmetric duplicates fold correctly.
    Other inputs are returned unchanged.

    Parameters
    ----------
    odfs : (n_voxels, n_amplitudes) array
    directions : (n_directions, 3) array
    """
    directions = np.asarray(directions, dtype=float)
    n_directions = len(directions)
    half = n_directions // 2
    full_sphere = (
        odfs.shape[1] == n_directions
        and n_directions % 2 == 0
        and np.allclose(directions[half:], -directions[:half], atol=1e-4)
    )
    if not full_sphere:
        return odfs, directions

    return np.maximum(odfs[:, :half], odfs[:, half:]), directions[:half]


def hemisphere_from_directions(directions, n_amplitudes):
    """Build the hemisphere whose vertices match the columns of an ODF amplitude array.

    Parameters
    ----------
    directions : (N, 3) array
        Sampling directions. Antipodal duplicates are removed, so a full sphere
        whose second half mirrors the first is accepted as well.
    n_amplitudes : int
        Number of ODF amplitudes per voxel.

    Returns
    -------
    dipy.core.sphere.HemiSphere
    """
    hemisphere = HemiSphere(xyz=np.asarray(directions, dtype=float))
    if len(hemisphere.vertices) != n_amplitudes:
        raise ValueError(
            f'The ODFs have {n_amplitudes} amplitudes per voxel, but the directions describe '
            f'{len(hemisphere.vertices)} unique hemisphere vertices.'
        )

    return hemisphere


def orient_faces_outward(vertices, faces):
    """Order each triangle's vertices counter-clockwise when seen from outside the sphere.

    TRXViz culls back faces, so triangles wound the other way are not drawn. DIPY's
    convex-hull faces have no consistent winding, and mirroring the hemisphere or rotating
    it with a reflecting affine flips more of them.
    """
    faces = np.array(faces, dtype=np.int64)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    inward = np.einsum('ij,ij->i', normals, a + b + c) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def voxel_to_world_rotation(affine):
    """Return the rotation (with reflections) part of an affine, without voxel scaling."""
    linear = np.asarray(affine, dtype=float)[:3, :3]
    return linear / np.linalg.norm(linear, axis=0)


def _to_world(vectors, rotation):
    world = np.asarray(vectors, dtype=float) @ rotation.T
    norms = np.linalg.norm(world, axis=-1, keepdims=True)
    # Keep the length of each vector (peak lengths encode relative peak size)
    in_norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(world * in_norms, norms, out=np.zeros_like(world), where=norms > 0)


def find_peaks(odfs, hemisphere, npeaks=3, relative_peak_threshold=0.1, min_separation_angle=15):
    """Find the peaks of each ODF.

    Parameters
    ----------
    odfs : (n_voxels, n_directions) array
    hemisphere : dipy.core.sphere.HemiSphere
    npeaks : int
        Maximum number of peaks kept per voxel.

    Returns
    -------
    counts : (n_voxels,) array of int
        Number of peaks in each voxel.
    directions : (sum(counts), 3) array
        Peak directions, scaled by the peak's amplitude relative to the largest peak in
        the voxel.
    """
    counts = np.zeros(len(odfs), dtype=int)
    directions = []
    for voxel, odf in enumerate(odfs):
        if not np.any(odf):
            continue

        dirs, values, _ = peak_directions(
            odf.astype(float),
            hemisphere,
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle,
        )
        n_found = min(npeaks, len(values))
        if n_found == 0 or values[0] <= 0:
            continue

        counts[voxel] = n_found
        directions.append(dirs[:n_found] * (values[:n_found] / values[0])[:, None])

    if directions:
        directions = np.concatenate(directions)
    else:
        directions = np.zeros((0, 3))

    return counts, directions


def write_odx(out_dir, affine, mask, odfs, hemisphere, peak_counts, peak_directions_vox):
    """Write ODFs and peaks to an ODX directory.

    Parameters
    ----------
    out_dir : str
        Path of the ODX directory to create.
    affine : (4, 4) array
        Voxel-to-RAS+ affine.
    mask : (X, Y, Z) bool array
        Voxels to write.
    odfs : (mask.sum(), n_directions) array
        ODF amplitudes of the voxels in ``mask``, in C order.
    hemisphere : dipy.core.sphere.HemiSphere
        Sampling directions of ``odfs``, along the voxel axes.
    peak_counts : (mask.sum(),) array of int
    peak_directions_vox : (peak_counts.sum(), 3) array
        Peak directions along the voxel axes.
    """
    rotation = voxel_to_world_rotation(affine)
    sphere = hemisphere.mirror()
    vertices = _to_world(sphere.vertices, rotation)
    faces = orient_faces_outward(vertices, sphere.faces)
    peak_dirs = _to_world(peak_directions_vox, rotation)
    n_voxels = int(mask.sum())
    offsets = np.concatenate([[0], np.cumsum(peak_counts)])

    os.makedirs(op.join(out_dir, 'sphere'), exist_ok=True)
    os.makedirs(op.join(out_dir, 'odf'), exist_ok=True)
    header = {
        'VOXEL_TO_RASMM': np.asarray(affine, dtype=float).tolist(),
        'DIMENSIONS': [int(dim) for dim in mask.shape],
        'NB_VOXELS': n_voxels,
        'NB_PEAKS': int(offsets[-1]),
        'NB_SPHERE_VERTICES': int(vertices.shape[0]),
        'NB_SPHERE_FACES': int(faces.shape[0]),
        # The amplitudes cover one hemisphere: the first half of the sphere's vertices
        'ODF_SAMPLE_DOMAIN': 'hemisphere',
    }
    with open(op.join(out_dir, 'header.json'), 'w') as fobj:
        json.dump(header, fobj, indent=2)

    mask.astype('<u1').tofile(op.join(out_dir, 'mask.uint8'))
    offsets.astype('<u4').tofile(op.join(out_dir, 'offsets.uint32'))
    peak_dirs.astype('<f4').tofile(op.join(out_dir, 'directions.3.float32'))
    vertices.astype('<f4').tofile(op.join(out_dir, 'sphere', 'vertices.3.float32'))
    faces.astype('<u4').tofile(op.join(out_dir, 'sphere', 'faces.3.uint32'))
    odfs = np.asarray(odfs, dtype='<f4')
    odfs.tofile(op.join(out_dir, 'odf', f'amplitudes.{odfs.shape[1]}.float32'))
    return out_dir


def _view_for_voxel_axis(affine, voxel_axis):
    """Name the TRXViz slice view that shows slices perpendicular to a voxel axis."""
    world_axis = int(nb.orientations.io_orientation(affine)[voxel_axis, 0])
    return _VIEW_NAMES[world_axis]


def background_for_display(data, mask):
    """Rescale a background image to [0, 1] with an S-shaped gray ramp.

    The 1st and 99.5th percentiles of the in-mask intensities map to 0 and 1, and the
    ramp in between follows VTK's default "S-curve" lookup table, which the FURY-based
    figures used. It adds contrast in the midtones, where white matter sits.
    """
    finite = np.nan_to_num(np.asarray(data, dtype='float32'))
    in_mask = mask & (finite > 0)
    values = finite[in_mask] if np.any(in_mask) else finite.ravel()
    low, high = np.percentile(values, [1, 99.5])
    if high <= low:
        return np.zeros_like(finite)

    scaled = np.clip((finite - low) / (high - low), 0, 1)
    return ((1 - np.cos(np.pi * scaled)) / 2).astype('float32')


def underlay_for_axis(display_img, ref_affine, ref_shape, voxel_axis):
    """Resample an underlay so it slices like the reference grid along one voxel axis.

    TRXViz draws every volume at the same slice index, so an underlay on another grid
    (e.g., a 1 mm anatomical image behind 2 mm ODFs) would show the wrong slice. The
    returned image keeps the reference grid's voxels along ``voxel_axis`` and divides the
    other two axes into voxels about as fine as the underlay's, so slice ``k`` is at the
    same position in both images while the underlay keeps its in-plane resolution.

    Parameters
    ----------
    display_img : nibabel image
        Underlay in [0, 1] (see :func:`background_for_display`), in the same world space
        as the reference grid.
    ref_affine : (4, 4) array
    ref_shape : tuple of 3 ints
    voxel_axis : int
    """
    from nibabel.processing import resample_from_to

    ref_zooms = np.sqrt((np.asarray(ref_affine)[:3, :3] ** 2).sum(0))
    finest = min(display_img.header.get_zooms()[:3])
    factors = np.maximum(np.round(ref_zooms / finest), 1).astype(int)
    factors[voxel_axis] = 1
    # Voxel i of the finer grid is centered at (i + 0.5) / f - 0.5 in reference voxels
    to_ref = np.diag(np.append(1 / factors, 1.0))
    to_ref[:3, 3] = 0.5 / factors - 0.5
    shape = tuple(int(n) for n in np.asarray(ref_shape[:3]) * factors)
    resampled = resample_from_to(display_img, (shape, ref_affine @ to_ref), order=1)
    data = np.clip(resampled.get_fdata(dtype='float32'), 0, 1)
    return nb.Nifti1Image(data, resampled.affine)


def background_window(brightness=BACKGROUND_BRIGHTNESS):
    """Return the TRXViz window (center, width) that shows a [0, 1] image at ``brightness``.

    TRXViz min-max normalizes the volume, which leaves a [0, 1] image unchanged, and then
    maps ``center - width / 2`` to black and ``center + width / 2`` to white.
    """
    width = 1 / brightness
    return width / 2, width


def decode_srgb(png_file):
    """Convert a PNG rendered into TRXViz's sRGB target back to linear values, in place.

    TRXViz 0.6.0 renders into an sRGB texture, which brightens and desaturates the
    shaders' colors (glyphs are colored by ``abs(direction)``, as in FURY). Decoding
    restores the colors that the shaders computed, as an ordinary target would store them.
    """
    with Image.open(png_file) as img:
        rgba = np.asarray(img.convert('RGBA'), dtype='float32') / 255

    rgb = rgba[..., :3]
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    rgba[..., :3] = linear
    Image.fromarray(np.round(rgba * 255).astype('uint8'), mode='RGBA').save(png_file)
    return png_file


def build_project(
    volume_file,
    odx_dir,
    affine,
    voxel_axis,
    center_ijk,
    half_extent_mm,
    layer,
    window=(0.5, 1.0),
    underlay_file=None,
):
    """Build a TRXViz project that shows one slice of the background with ODFs or peaks.

    Parameters
    ----------
    volume_file : str
        Background NIfTI. It must share the ODX's voxel grid.
    odx_dir : str
    affine : (4, 4) array
        Voxel-to-RAS+ affine of the background and the ODX.
    voxel_axis : int
        Voxel axis perpendicular to the slice.
    center_ijk : sequence of 3 floats
        Voxel coordinates of the view center. Its ``voxel_axis`` entry selects the slice.
    half_extent_mm : float
        Half the height of the view, in mm.
    layer : {'odfs', 'peaks'}
    window : tuple of 2 floats
        Background intensity window (center, width) in normalized units.
    underlay_file : str, optional
        Image to show instead of ``volume_file``, such as the anatomical image. It must
        slice like ``volume_file`` along ``voxel_axis`` (see :func:`underlay_for_axis`),
        because TRXViz draws every volume at the same slice index. ``volume_file`` is still
        loaded first, without being shown, so the slice index matches the ODX grid.

    Notes
    -----
    TRXViz draws each ODF vertex at ``amplitude * min(voxel size)`` mm from the voxel
    center, so the ODX amplitudes should already be scaled for display (see
    :func:`scale_odfs_for_display`). TRXViz's own per-voxel normalization is left off
    because, in TRXViz 0.6.0, it scales glyphs to about twice the voxel size.
    """
    view = _view_for_voxel_axis(affine, voxel_axis)
    center_ras = nb.affines.apply_affine(affine, center_ijk)
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))

    nodes = {
        1: {'VolumeSource': {'source_id': 0 if underlay_file is None else 2}},
        2: {
            'VolumeDisplay': {
                'colormap': 'Grayscale',
                'opacity': 1.0,
                'window_center': float(window[0]),
                'window_width': float(window[1]),
            }
        },
        3: {'OdxSource': {'source_id': 1}},
    }
    wires = [((1, 0), (2, 0))]
    if layer == 'odfs':
        nodes[4] = {
            'OdfGlyphRenderer': {
                'slice_axis': view,
                'subtract_iso': False,
                'norm_within_voxel': False,
                'visible': True,
                'detail': 3,
            }
        }
        # OdxSource output 1 is the ODF field
        wires.append(((3, 1), (4, 0)))
    elif layer == 'peaks':
        nodes[4] = {
            'Fixel2DDisplay': {
                'line_width': 0.012,
                'opacity': 1.0,
                'slab_thickness_mm': float(zooms[voxel_axis]),
                # Each half of a fixel line is length_scale * min(voxel size) long
                'length_scale': GLYPH_RADIUS,
                'visible': True,
                'auto_gate_from_otsu': False,
                # Without scalars, the default gate hides every fixel; draw them all
                'opacity_gate': {'range': [0.0, 0.0], 'below': 1.0, 'above': 1.0},
            }
        }
        # OdxSource output 0 is the fixels
        wires.append(((3, 0), (4, 0)))
    else:
        raise ValueError(f'Unknown layer: {layer}')

    cameras = []
    for view_name in _VIEW_ORDER:
        horizontal, vertical = _IN_PLANE_AXES[view_name]
        cameras.append(
            {
                'center': [float(center_ras[horizontal]), float(center_ras[vertical])],
                'half_extent': float(half_extent_mm),
                'rotation': 0.0,
            }
        )

    # TRXViz converts slice positions to indices with the first volume, so it must be the
    # one that shares the ODX's grid
    assets = [
        {'Volume': {'id': 0, 'path': op.abspath(volume_file)}},
        {'Odx': {'id': 1, 'path': op.abspath(odx_dir)}},
    ]
    if underlay_file is not None:
        assets.append({'Volume': {'id': 2, 'path': op.abspath(underlay_file)}})

    return {
        'version': 1,
        'document': {
            'next_node_uuid': max(nodes) + 1,
            'assets': assets,
            'graph': {
                'nodes': {
                    str(uuid): {'node': {'uuid': uuid, 'label': next(iter(op_)), 'op': op_}}
                    for uuid, op_ in nodes.items()
                },
                'wires': [
                    {'from': {'node': src, 'output': out}, 'to': {'node': dst, 'input': in_}}
                    for (src, out), (dst, in_) in wires
                ],
            },
            'slice_view_3d': {
                'visible': [name == view for name in _VIEW_ORDER],
                # Axial (z), coronal (y), and sagittal (x) slice positions in RAS+ mm
                'positions_ras': [float(center_ras[axis]) for axis in (2, 1, 0)],
            },
            'slice_view_ui': {
                'mode': 'Slice',
                'single_view': view,
                'lightbox_axis': view,
                'lightbox_rows': 1,
                'lightbox_cols': 1,
                'active_axis': _VIEW_ORDER.index(view),
                'ortho_show_row': False,
                'slice_cameras': cameras,
            },
        },
    }


def render_project(project, project_file, png_file, size):
    """Write a TRXViz project and render its 2D view to a square PNG."""
    with open(project_file, 'w') as fobj:
        json.dump(project, fobj, indent=1)

    cmd = [
        find_trxviz_cli(),
        'render',
        '--project',
        project_file,
        '--view',
        '2d',
        '--width',
        str(size),
        '--height',
        str(size),
        '--out',
        png_file,
    ]
    LOGGER.info('Running %s', ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not op.exists(png_file):
        raise RuntimeError(
            TRXVIZ_ERROR.format(
                png=png_file,
                cmd=' '.join(cmd),
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        )

    return decode_srgb(png_file)


def _slice_mask(shape, voxel_axis, index):
    mask = np.zeros(shape, dtype=bool)
    slicer = [slice(None)] * 3
    slicer[voxel_axis] = index
    mask[tuple(slicer)] = True
    return mask


def _in_plane_half_extent(affine, voxel_axis, ijk_min, ijk_max):
    """Half the larger in-plane extent, in mm, of a box of voxels."""
    corners = np.array(
        [[(ijk_min, ijk_max)[bit >> axis & 1][axis] for axis in range(3)] for bit in range(8)],
        dtype=float,
    )
    corners_ras = nb.affines.apply_affine(affine, corners)
    view = _view_for_voxel_axis(affine, voxel_axis)
    spans = np.ptp(corners_ras, axis=0)
    return float(max(spans[axis] for axis in _IN_PLANE_AXES[view]) / 2)


def _roi_bbox(roi_data, roi_id):
    """Return the centroid and the voxel bounding box of an ROI."""
    coords = np.argwhere(roi_data == roi_id)
    if coords.size == 0:
        raise ValueError(f'ROI {roi_id} is empty.')

    return coords.mean(0), coords.min(0), coords.max(0)


def _stitch(tiles, n_rows, n_cols, tile_size, out_file):
    """Paste tiles (listed column by column) into one image."""
    mosaic = Image.new('RGB', (tile_size * n_cols, tile_size * n_rows))
    for tile_num, tile in enumerate(tiles):
        col, row = divmod(tile_num, n_rows)
        with Image.open(tile) as tile_img:
            mosaic.paste(tile_img.convert('RGB'), (col * tile_size, row * tile_size))

    mosaic.save(out_file)
    return out_file


def plot_odfs_and_peaks(
    odf_img,
    directions,
    background_img,
    peaks_file,
    mask_img=None,
    odf_rois_img=None,
    odfs_file=None,
    subtract_iso=False,
    n_cuts=3,
    padding=10,
    tile_size=800,
    workdir='.',
    anat_img=None,
    anat_mask_img=None,
):
    """Plot a mosaic of peak directions and, optionally, of ODF glyphs in three ROIs.

    Parameters
    ----------
    odf_img : nibabel image
        4D image of ODF amplitudes.
    directions : (N, 3) array
        Sampling directions of the ODF amplitudes, along the voxel axes.
    background_img : nibabel image
        Background image on the ODFs' voxel grid.
    peaks_file : str
        Output PNG with ``n_cuts`` axial, sagittal, and coronal slices (one column each).
    mask_img : nibabel image, optional
        Voxels in which peaks and ODFs are drawn.
    odf_rois_img : nibabel image, optional
        Labels 1, 2, and 3 mark the centrum semiovale (shown in an axial slice) and
        two regions shown in coronal slices.
    odfs_file : str, optional
        Output PNG for the ODF glyphs. Required with ``odf_rois_img``.
    subtract_iso : bool
        Subtract each ODF's minimum before drawing its glyph.
    anat_img : nibabel image, optional
        Anatomical image in the same world space as the ODFs (e.g., QSIPrep's AC-PC
        T1w or T2w). If given, it is shown behind the glyphs instead of the background.
    anat_mask_img : nibabel image, optional
        Brain mask of ``anat_img``, used to set its intensity range.
    """
    os.makedirs(workdir, exist_ok=True)
    shape = odf_img.shape[:3]
    if background_img.shape[:3] != shape:
        raise ValueError(
            f'The background image ({background_img.shape[:3]}) and the ODFs ({shape}) '
            'must share a voxel grid.'
        )

    affine = odf_img.affine
    background = np.asarray(background_img.dataobj, dtype='float32')
    if mask_img is None:
        LOGGER.info('No mask image for plotting peaks')
        mask = np.ones(shape, dtype=bool)
    else:
        mask = np.asanyarray(mask_img.dataobj) > 0

    # The background is written next to the project so TRXViz reads it from the same grid
    background_file = op.abspath(op.join(workdir, 'background.nii.gz'))
    nb.Nifti1Image(background_for_display(background, mask), affine).to_filename(background_file)

    underlay_files = {}
    if anat_img is not None:
        anat_mask = (
            np.asanyarray(anat_mask_img.dataobj) > 0
            if anat_mask_img is not None
            else np.ones(anat_img.shape[:3], dtype=bool)
        )
        anat_display = nb.Nifti1Image(
            background_for_display(anat_img.get_fdata(dtype='float32'), anat_mask),
            anat_img.affine,
        )
        for voxel_axis in range(3):
            underlay_file = op.abspath(op.join(workdir, f'underlay_axis-{voxel_axis}.nii.gz'))
            underlay_for_axis(anat_display, affine, shape, voxel_axis).to_filename(underlay_file)
            underlay_files[voxel_axis] = underlay_file

    # Plan the views: (name, voxel axis, view center in voxels, half extent in mm, layer)
    views = []
    slice_indices = slices_from_bbox(background, cuts=n_cuts, padding=padding)
    LOGGER.info('Plotting slice indices %s', slice_indices)
    bbox_min = np.argwhere(background > 0).min(0)
    bbox_max = np.argwhere(background > 0).max(0)
    volume_center = (bbox_min + bbox_max) / 2
    for axis_name, voxel_axis in (('z', 2), ('x', 0), ('y', 1)):
        half_extent = 1.05 * _in_plane_half_extent(affine, voxel_axis, bbox_min, bbox_max)
        for index in slice_indices[axis_name]:
            center = volume_center.copy()
            center[voxel_axis] = index
            views.append((f'peaks_{axis_name}{index:03d}', voxel_axis, center, half_extent))

    roi_views = []
    if odf_rois_img is not None:
        roi_data = np.asanyarray(odf_rois_img.dataobj)
        rois = []
        for roi_id, voxel_axis in ((1, 2), (2, 1), (3, 1)):
            centroid, roi_min, roi_max = _roi_bbox(roi_data, roi_id)
            half_extent = _in_plane_half_extent(affine, voxel_axis, roi_min, roi_max)
            rois.append((roi_id, voxel_axis, np.round(centroid), half_extent))

        # All three views share one zoom, as in the previous FURY-based figure
        half_extent = max(1.7 * max(roi[3] for roi in rois), 10.0)
        roi_views = [
            (f'odfs_roi{roi_id}', voxel_axis, centroid, half_extent)
            for roi_id, voxel_axis, centroid, _ in rois
        ]

    # Only the voxels on the plotted slices go into the ODX
    odx_mask = np.zeros(shape, dtype=bool)
    for _, voxel_axis, center, _ in views + roi_views:
        odx_mask |= _slice_mask(shape, voxel_axis, round(center[voxel_axis]))
    odx_mask &= mask

    odfs = np.nan_to_num(odf_img.get_fdata(dtype='float32')[odx_mask])

    odfs, directions = fold_full_sphere(odfs, directions)
    hemisphere = hemisphere_from_directions(directions, odfs.shape[1])
    peak_counts, peak_dirs = find_peaks(odfs, hemisphere)
    odx_dir = write_odx(
        op.join(workdir, 'odfs.odx'),
        affine,
        odx_mask,
        scale_odfs_for_display(odfs, subtract_iso=subtract_iso),
        hemisphere,
        peak_counts,
        peak_dirs,
    )

    def _render(name, voxel_axis, center, half_extent, layer):
        project = build_project(
            background_file,
            odx_dir,
            affine,
            voxel_axis,
            center,
            half_extent,
            layer,
            window=background_window(),
            underlay_file=underlay_files.get(voxel_axis),
        )
        return render_project(
            project,
            op.join(workdir, f'{name}.json'),
            op.join(workdir, f'{name}.png'),
            tile_size,
        )

    peak_tiles = [_render(*view, 'peaks') for view in views]
    _stitch(peak_tiles, n_cuts, 3, tile_size, peaks_file)

    if roi_views:
        if odfs_file is None:
            raise ValueError('odfs_file is required to plot ODFs in ROIs.')

        odf_tiles = [_render(*view, 'odfs') for view in roi_views]
        _stitch(odf_tiles, 1, 3, tile_size, odfs_file)

    return peaks_file, odfs_file
