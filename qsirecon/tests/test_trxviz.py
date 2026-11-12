"""Tests for the TRXViz-based ODF and peak report figures."""

import json
import shutil

import nibabel as nb
import numpy as np
import pytest
from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor_odf
from PIL import Image

from qsirecon.interfaces.reports import ReconPeaksReport
from qsirecon.viz import trxviz

# QSIPrep writes LPS+ images
LPS_AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 30.0],
        [0.0, -2.0, 0.0, 30.0],
        [0.0, 0.0, 2.0, -10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

needs_trxviz = pytest.mark.skipif(
    shutil.which('trxviz-cli') is None,
    reason='trxviz-cli is not installed',
)


def _hemisphere():
    sphere = get_sphere(name='repulsion724')
    return HemiSphere(xyz=sphere.vertices[sphere.vertices[:, 2] >= 0])


def _crossing_odf(hemisphere, angles):
    mevals = np.array([[0.0015, 0.0003, 0.0003]] * 2)
    return multi_tensor_odf(hemisphere.vertices, mevals, angles, [50, 50])


def _synthetic_dataset(shape=(16, 16, 10)):
    """ODFs with x-oriented fibers in the first half of the grid and an x/y crossing after."""
    hemisphere = _hemisphere()
    single = _crossing_odf(hemisphere, [(90, 0), (90, 0)])
    crossing = _crossing_odf(hemisphere, [(90, 0), (90, 90)])
    odfs = np.zeros(shape + (len(hemisphere.vertices),), dtype='float32')
    odfs[: shape[0] // 2] = single
    odfs[shape[0] // 2 :] = crossing
    mask = np.zeros(shape, dtype='uint8')
    mask[2:-2, 2:-2, 1:-1] = 1
    odfs *= mask[..., None]
    rois = np.zeros(shape, dtype='int16')
    rois[6:10, 6:10, 4:6] = 1
    rois[3:6, 7:9, 3:7] = 2
    rois[10:13, 7:9, 3:7] = 3
    return {
        'odf': nb.Nifti1Image(odfs, LPS_AFFINE),
        'directions': hemisphere.vertices,
        'background': nb.Nifti1Image(mask * 100.0, LPS_AFFINE),
        'mask': nb.Nifti1Image(mask, LPS_AFFINE),
        'rois': nb.Nifti1Image(rois, LPS_AFFINE),
    }


def test_hemisphere_from_directions_accepts_full_sphere():
    """A full sphere whose second half mirrors the first reduces to its first half."""
    hemisphere = _hemisphere()
    full = np.vstack([hemisphere.vertices, -hemisphere.vertices])
    result = trxviz.hemisphere_from_directions(full, len(hemisphere.vertices))
    assert len(result.vertices) == len(hemisphere.vertices)

    with pytest.raises(ValueError, match='unique hemisphere vertices'):
        trxviz.hemisphere_from_directions(full, len(full))


def test_fold_full_sphere():
    """AMICO-style full-sphere ODFs fold onto the hemisphere; others pass through."""
    hemisphere = _hemisphere()
    n = len(hemisphere.vertices)
    full = np.vstack([hemisphere.vertices, -hemisphere.vertices])
    # Peaks only in the first half, as peaks_to_odfs writes them
    odfs = np.zeros((2, 2 * n))
    odfs[0, 5] = 0.7
    odfs[1, n + 9] = 0.4  # and one mirrored peak, as a symmetric layout would have
    folded, directions = trxviz.fold_full_sphere(odfs, full)
    assert folded.shape == (2, n)
    np.testing.assert_allclose(directions, hemisphere.vertices)
    assert folded[0, 5] == 0.7
    assert folded[1, 9] == 0.4
    # The folded result matches a hemisphere of the right size
    assert len(trxviz.hemisphere_from_directions(directions, folded.shape[1]).vertices) == n

    # Hemisphere-sampled ODFs (DSI Studio GQI, MRtrix, DIPY) are left alone
    half_odfs = np.ones((2, n))
    same, same_dirs = trxviz.fold_full_sphere(half_odfs, full)
    assert same is half_odfs
    assert len(same_dirs) == 2 * n


def test_find_peaks_crossing():
    """Two orthogonal fibers give two peaks, scaled relative to the largest one."""
    hemisphere = _hemisphere()
    odfs = np.stack(
        [
            _crossing_odf(hemisphere, [(90, 0), (90, 90)]),
            np.zeros(len(hemisphere.vertices)),
        ]
    )
    counts, directions = trxviz.find_peaks(odfs, hemisphere)
    assert counts.tolist() == [2, 0]
    assert directions.shape == (2, 3)
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    # One peak along x, one along y (up to sign)
    np.testing.assert_allclose(np.sort(np.abs(unit).max(axis=0))[1:], [1, 1], atol=0.05)
    assert np.isclose(np.linalg.norm(directions, axis=1).max(), 1)


def test_scale_odfs_for_display():
    odfs = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [2.0, 2.0, 4.0]])
    scaled = trxviz.scale_odfs_for_display(odfs)
    np.testing.assert_allclose(scaled.max(axis=1), [trxviz.GLYPH_RADIUS, 0, trxviz.GLYPH_RADIUS])
    np.testing.assert_allclose(scaled[0], np.array([1, 2, 3]) / 3 * trxviz.GLYPH_RADIUS)

    subtracted = trxviz.scale_odfs_for_display(odfs, subtract_iso=True)
    np.testing.assert_allclose(subtracted[0], [0, 0.5 * trxviz.GLYPH_RADIUS, trxviz.GLYPH_RADIUS])


def test_background_for_display():
    """The background spans [0, 1] with an S-shaped ramp between the robust limits."""
    data = np.zeros((10, 10, 10), dtype='float32')
    data[2:8, 2:8, 2:8] = np.linspace(1, 100, 216).reshape(6, 6, 6)
    mask = data > 0
    display = trxviz.background_for_display(data, mask)
    assert display.min() == 0
    assert display.max() == 1
    low, high = np.percentile(data[mask], [1, 99.5])
    scaled = np.clip((data - low) / (high - low), 0, 1)
    np.testing.assert_allclose(display, (1 - np.cos(np.pi * scaled)) / 2, atol=1e-6)
    # The S-curve darkens the lower half of the ramp and brightens the upper half
    lower = (scaled > 0) & (scaled < 0.5)
    upper = (scaled > 0.5) & (scaled < 1)
    assert lower.any()
    assert upper.any()
    assert np.all(display[lower] < scaled[lower])
    assert np.all(display[upper] > scaled[upper])

    center, width = trxviz.background_window(0.5)
    assert (center, width) == (1.0, 2.0)


def test_underlay_for_axis():
    """A finer underlay keeps the reference slicing along the chosen axis."""
    ref_shape = (10, 12, 8)
    anat_affine = LPS_AFFINE.copy()
    anat_affine[:3, :3] /= 2  # 1 mm voxels in the same world space
    anat = nb.Nifti1Image(np.random.default_rng(0).random((20, 24, 16)), anat_affine)

    underlay = trxviz.underlay_for_axis(anat, LPS_AFFINE, ref_shape, voxel_axis=2)
    assert underlay.shape == (20, 24, 8)
    # Slice k along z is at the same world position in both grids
    for k in (0, 3, 7):
        np.testing.assert_allclose(
            nb.affines.apply_affine(underlay.affine, [0, 0, k])[2],
            nb.affines.apply_affine(LPS_AFFINE, [0, 0, k])[2],
        )

    # In-plane voxels subdivide the reference voxels, centered within them
    np.testing.assert_allclose(
        nb.affines.apply_affine(underlay.affine, [0.5, 0.5, 0])[:2],
        nb.affines.apply_affine(LPS_AFFINE, [0, 0, 0])[:2],
    )
    assert 0 <= underlay.get_fdata().min() <= underlay.get_fdata().max() <= 1


def test_decode_srgb(tmp_path):
    """sRGB-encoded pixels are converted back to linear values."""
    png = tmp_path / 'tile.png'
    encoded = np.array([[[0, 0, 0], [255, 255, 255], [188, 128, 54]]], dtype='uint8')
    Image.fromarray(encoded).save(png)
    trxviz.decode_srgb(str(png))
    with Image.open(png) as img:
        decoded = np.asarray(img.convert('RGB'))

    np.testing.assert_array_equal(decoded[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(decoded[0, 1], [255, 255, 255])
    # sRGB 188/128/54 is about 50%/22%/4% linear
    np.testing.assert_allclose(decoded[0, 2], [128, 56, 9], atol=2)


def test_write_odx(tmp_path):
    """The ODX holds the masked voxels in C order and RAS+ directions."""
    hemisphere = _hemisphere()
    shape = (3, 4, 5)
    mask = np.zeros(shape, dtype=bool)
    mask[1, 2, 3] = True
    mask[2, 0, 1] = True
    odfs = np.ones((2, len(hemisphere.vertices)), dtype='float32')
    peak_counts = np.array([2, 1])
    peaks_vox = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])

    odx = trxviz.write_odx(
        str(tmp_path / 'test.odx'),
        LPS_AFFINE,
        mask,
        odfs,
        hemisphere,
        peak_counts,
        peaks_vox,
    )

    header = json.loads((tmp_path / 'test.odx' / 'header.json').read_text())
    n_vertices = 2 * len(hemisphere.vertices)
    assert header['DIMENSIONS'] == list(shape)
    assert header['NB_VOXELS'] == 2
    assert header['NB_PEAKS'] == 3
    assert header['NB_SPHERE_VERTICES'] == n_vertices
    assert header['ODF_SAMPLE_DOMAIN'] == 'hemisphere'
    np.testing.assert_allclose(header['VOXEL_TO_RASMM'], LPS_AFFINE)

    written_mask = np.fromfile(f'{odx}/mask.uint8', dtype='<u1').reshape(shape)
    np.testing.assert_array_equal(written_mask, mask)
    np.testing.assert_array_equal(np.fromfile(f'{odx}/offsets.uint32', dtype='<u4'), [0, 2, 3])

    # Every face is wound counter-clockwise seen from outside, so none is culled
    faces_check = np.fromfile(f'{odx}/sphere/faces.3.uint32', dtype='<u4').reshape(-1, 3)
    verts_check = np.fromfile(f'{odx}/sphere/vertices.3.float32', dtype='<f4').reshape(-1, 3)
    a, b, c = (verts_check[faces_check[:, i]] for i in range(3))
    assert np.all(np.einsum('ij,ij->i', np.cross(b - a, c - a), a + b + c) > 0)

    # LPS voxel axes: x and y flip sign in RAS+; peak lengths are kept
    directions = np.fromfile(f'{odx}/directions.3.float32', dtype='<f4').reshape(-1, 3)
    np.testing.assert_allclose(directions, [[-1, 0, 0], [0, -0.5, 0], [0, 0, 1]], atol=1e-6)

    vertices = np.fromfile(f'{odx}/sphere/vertices.3.float32', dtype='<f4').reshape(-1, 3)
    assert vertices.shape == (n_vertices, 3)
    # The second half of the sphere mirrors the first, as ODX expects for hemisphere ODFs
    np.testing.assert_allclose(
        vertices[n_vertices // 2 :], -vertices[: n_vertices // 2], atol=1e-6
    )
    faces = np.fromfile(f'{odx}/sphere/faces.3.uint32', dtype='<u4').reshape(-1, 3)
    assert faces.shape[0] == header['NB_SPHERE_FACES']
    assert faces.max() < n_vertices

    amplitudes = np.fromfile(
        f'{odx}/odf/amplitudes.{len(hemisphere.vertices)}.float32',
        dtype='<f4',
    )
    assert amplitudes.size == odfs.size


def test_build_project_views():
    """Voxel axes map to TRXViz views and RAS+ slice positions."""
    center = np.array([5.0, 6.0, 7.0])
    center_ras = nb.affines.apply_affine(LPS_AFFINE, center)
    expected = {2: 'Axial', 1: 'Coronal', 0: 'Sagittal'}
    for voxel_axis, view in expected.items():
        project = trxviz.build_project(
            'bg.nii.gz',
            'odfs.odx',
            LPS_AFFINE,
            voxel_axis,
            center,
            20.0,
            'peaks' if voxel_axis else 'odfs',
        )
        document = project['document']
        assert document['slice_view_ui']['single_view'] == view
        assert document['slice_view_3d']['visible'] == [
            name == view for name in ('Axial', 'Coronal', 'Sagittal')
        ]
        np.testing.assert_allclose(
            document['slice_view_3d']['positions_ras'],
            center_ras[[2, 1, 0]],
        )
        ops = [node['node']['op'] for node in document['graph']['nodes'].values()]
        op_names = {next(iter(op)) for op in ops}
        assert ('Fixel2DDisplay' if voxel_axis else 'OdfGlyphRenderer') in op_names

    with pytest.raises(ValueError, match='Unknown layer'):
        trxviz.build_project('bg.nii.gz', 'odfs.odx', LPS_AFFINE, 0, center, 20.0, 'glyphs')


def test_missing_trxviz_cli(monkeypatch):
    monkeypatch.setattr(trxviz.shutil, 'which', lambda _: None)
    with pytest.raises(FileNotFoundError, match='skip-odf-reports'):
        trxviz.find_trxviz_cli()


@needs_trxviz
def test_recon_peaks_report(tmp_path):
    """Render both reports with trxviz-cli and check that peaks were drawn."""
    data = _synthetic_dataset()
    for name in ('odf', 'background', 'mask', 'rois'):
        data[name].to_filename(tmp_path / f'{name}.nii.gz')
    np.save(tmp_path / 'directions.npy', data['directions'])

    results = ReconPeaksReport(
        odf_file=str(tmp_path / 'odf.nii.gz'),
        directions_file=str(tmp_path / 'directions.npy'),
        background_image=str(tmp_path / 'background.nii.gz'),
        mask_file=str(tmp_path / 'mask.nii.gz'),
        odf_rois=str(tmp_path / 'rois.nii.gz'),
        padding=1,
        tile_size=200,
    ).run(cwd=str(tmp_path))

    with Image.open(results.outputs.peak_report) as peaks_img:
        assert peaks_img.size == (600, 600)
        peaks = np.asarray(peaks_img.convert('RGB'), dtype=int)

    with Image.open(results.outputs.odf_report) as odfs_img:
        assert odfs_img.size == (600, 200)
        odfs = np.asarray(odfs_img.convert('RGB'), dtype=int)

    # The background is grayscale, so colored pixels come from peaks and ODF glyphs
    for image in (peaks, odfs):
        saturation = image.max(axis=2) - image.min(axis=2)
        assert (saturation > 100).mean() > 0.001
