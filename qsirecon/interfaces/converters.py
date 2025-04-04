"""Handle merging and spliting of DSI files."""

import gzip
import logging
import os
import os.path as op
import re
import shutil
import subprocess
from pathlib import Path

import nibabel as nb
import numpy as np

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO

from dipy.core.geometry import cart2sphere
from dipy.core.sphere import HemiSphere
from dipy.direction import peak_directions
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from pkg_resources import resource_filename as pkgr
from scipy.io.matlab import loadmat, savemat

LOGGER = logging.getLogger("nipype.workflow")
ODF_COLS = 20000  # Number of columns in DSI Studio odf split
MIN_NONZERO = 1e-6


class FODtoFIBGZInputSpec(BaseInterfaceInputSpec):
    mif_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    num_fibers = traits.Int(5, usedefault=True)
    unit_odf = traits.Bool(False, usedefault=True)
    output_fib_file = File()


class FODtoFIBGZOutputSpec(TraitedSpec):
    fib_file = File(exists=True)


class FODtoFIBGZ(SimpleInterface):
    input_spec = FODtoFIBGZInputSpec
    output_spec = FODtoFIBGZOutputSpec

    def _run_interface(self, runtime):
        mif_file = self.inputs.mif_file
        mask_file = self.inputs.mask_file
        if isdefined(self.inputs.output_fib_file):
            output_fib_path = Path(self.inputs.output_fib_file)
            if output_fib_path.name.endswith(".gz"):
                LOGGER.warning("A non-gzipped output will be written.")
                output_fib_path.name = output_fib_path.name[:-3]
            if not output_fib_path.is_absolute():
                output_fib_file = str(Path(runtime.cwd) / output_fib_path)

        else:
            output_fib_file = fname_presuffix(
                mif_file, newpath=runtime.cwd, suffix=".fib", use_ext=False
            )

        verts, faces = get_dsi_studio_ODF_geometry("odf8")
        num_dirs, _ = verts.shape
        hemisphere = num_dirs // 2
        x, y, z = verts[:hemisphere].T
        _, theta, phi = cart2sphere(x, y, -z)
        dirs_txt = op.join(runtime.cwd, "directions.txt")
        np.savetxt(dirs_txt, np.column_stack([phi, theta]))

        odf_amplitudes_nii = op.join(runtime.cwd, "amplitudes.nii")
        popen_run(["sh2amp", "-quiet", "-nonnegative", mif_file, dirs_txt, odf_amplitudes_nii])

        if not op.exists(odf_amplitudes_nii):
            raise FileNotFoundError("Unable to create %s", odf_amplitudes_nii)
        amplitudes_img = nb.load(odf_amplitudes_nii)

        if isdefined(mask_file):
            mask_img = nb.load(mask_file)
        else:
            ampl_data = amplitudes_img.get_fdata()
            ampl_mask = ampl_data.sum(3) > 1e-6
            mask_img = nb.Nifti1Image(ampl_mask.astype(float), amplitudes_img.affine)

        self._results["fib_file"] = output_fib_file
        amplitudes_to_fibgz(
            amplitudes_img,
            verts,
            faces,
            output_fib_file,
            mask_img,
            num_fibers=self.inputs.num_fibers,
            unit_odf=self.inputs.unit_odf,
        )
        os.remove(odf_amplitudes_nii)
        return runtime


class FIBGZtoFODInputSpec(BaseInterfaceInputSpec):
    fib_file = File(exists=True, mandatory=True)
    ref_image = File(exists=True, mandatory=True)
    subtract_iso = traits.Bool(True, usedefault=True)
    mif_file = File()


class FIBGZtoFODOutputSpec(TraitedSpec):
    mif_file = File(exists=True)


class FIBGZtoFOD(SimpleInterface):
    input_spec = FIBGZtoFODInputSpec
    output_spec = FIBGZtoFODOutputSpec

    def _run_interface(self, runtime):
        fib_file = self.inputs.fib_file
        if isdefined(self.inputs.mif_file):
            output_mif_file = self.inputs.mif_file
        else:
            output_mif_file = fname_presuffix(
                fib_file, newpath=runtime.cwd, suffix=".mif", use_ext=False
            )
        # Get the amplitudes as a Nifti1Image and sphere directions
        amplitudes, directions = fib2amps(
            fib_file, self.inputs.ref_image, self.inputs.subtract_iso
        )
        # convert them to MRTrix mif format
        amplitudes_to_sh_mif(amplitudes, directions, output_mif_file, runtime.cwd)
        self._results["mif_file"] = output_mif_file
        return runtime


class NODDItoFIBGZInputSpec(BaseInterfaceInputSpec):
    icvf_file = File(exists=True)
    isovf_file = File(exists=True)
    od_file = File(exists=True)
    modulated_icvf_file = File(exists=True)
    modulated_od_file = File(exists=True)
    directions_file = File(exists=True)
    mask_file = File(exists=True)


class NODDItoFIBGZOutputSpec(TraitedSpec):
    fibgz_file = File(exists=True)


class NODDItoFIBGZ(SimpleInterface):
    input_spec = NODDItoFIBGZInputSpec
    output_spec = NODDItoFIBGZOutputSpec

    def _run_interface(self, runtime):
        output_file = fname_presuffix(
            self.inputs.icvf_file, use_ext=False, newpath=runtime.cwd, suffix=".fib"
        )
        verts, faces = get_dsi_studio_ODF_geometry("odf8")
        amico_directions_to_fibgz(
            directions_img=nb.load(self.inputs.directions_file),
            od_img=nb.load(self.inputs.od_file),
            icvf_img=nb.load(self.inputs.icvf_file),
            modulated_od_img=nb.load(self.inputs.modulated_od_file),
            modulated_icvf_img=nb.load(self.inputs.modulated_icvf_file),
            isovf_img=nb.load(self.inputs.isovf_file),
            odf_dirs=verts,
            odf_faces=faces,
            output_file=output_file,
            mask_img=nb.load(self.inputs.mask_file),
        )
        self._results["fibgz_file"] = output_file

        return runtime


class _DSIStudioTrkToTckInputSpec(BaseInterfaceInputSpec):
    trk_file = File(exists=True, mandatory=True)
    reference_nifti = File(exists=True, mandatory=True)


class _DSIStudioTrkToTckOutputSpec(TraitedSpec):
    tck_file = File()


class DSIStudioTrkToTck(SimpleInterface):
    input_spec = _DSIStudioTrkToTckInputSpec
    output_spec = _DSIStudioTrkToTckOutputSpec

    def _run_interface(self, runtime):

        if self.inputs.trk_file.endswith(".gz"):
            with gzip.open(self.inputs.trk_file, "r") as trkf:
                dsi_trk = nb.streamlines.load(trkf)
        else:
            dsi_trk = nb.streamlines.load(self.inputs.trk_file)

        # load preprocessed dwi image
        dwi_img = nb.load(self.inputs.reference_nifti)

        # convert to voxel coordinates
        pts = dsi_trk.streamlines._data
        zooms = np.abs(np.diag(dsi_trk.header["voxel_to_rasmm"])[:3])
        voxel_coords = pts / zooms
        voxel_coords[:, 0] = dwi_img.shape[0] - voxel_coords[:, 0]
        voxel_coords[:, 1] = dwi_img.shape[1] - voxel_coords[:, 1]

        # create new tck
        new_data = nb.affines.apply_affine(dwi_img.affine, voxel_coords)
        dsi_trk.tractogram.streamlines._data = new_data
        tck = nb.streamlines.TckFile(dsi_trk.tractogram)
        tck_file = fname_presuffix(
            self.inputs.trk_file.rstrip(".gz"), newpath=runtime.cwd, use_ext=False, suffix=".tck"
        )
        tck.save(tck_file)
        self._results["tck_file"] = tck_file
        return runtime


class _MergeFODGQIFibsInputSpec(BaseInterfaceInputSpec):
    csd_fib_file = File(exists=True, mandatory=True)
    reference_fib_file = File(exists=True, mandatory=True)
    fibgz_map = File(exists=True)


class _MergeFODGQIFibsOutputSpec(TraitedSpec):
    fibgz = File(exists=True, mandatory=True)
    fibgz_map = File(exists=True)


class MergeFODGQIFibs(SimpleInterface):
    """Merge FOD and GQI fib files."""

    input_spec = _MergeFODGQIFibsInputSpec
    output_spec = _MergeFODGQIFibsOutputSpec

    def _run_interface(self, runtime):

        # fname presuffix doesn't work with .fib.gz
        fib_name = Path(self.inputs.reference_fib_file).name.replace(".odf.", ".odf.FOD.")
        merged_fib_file = str(Path(runtime.cwd) / fib_name)
        merged_fib_file = (
            merged_fib_file if not merged_fib_file.endswith(".gz") else merged_fib_file[:-3]
        )

        combine_gqi_and_csd_fib_files(
            path_gqi_fib=self.inputs.reference_fib_file,
            path_fod_fib=self.inputs.csd_fib_file,
            merged_fib=merged_fib_file,
        )

        # gzip the merged file
        merged_fibgz_file = merged_fib_file + ".gz"

        p = subprocess.Popen(
            ["gzip", "-v", merged_fib_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        response = p.communicate()
        if not p.returncode == 0:
            raise Exception(f"Gzip exitted with code {p.returncode}: {response}")
        if not Path(merged_fibgz_file).exists():
            raise Exception(f"Failed to gzip {merged_fib_file}")
        self._results["fibgz"] = merged_fibgz_file

        # Handle the map file if it was provided
        if isdefined(self.inputs.fibgz_map):
            LOGGER.info(f"Creating new map file to match {merged_fib_file}.")
            # DSI Studio stores the template of the mapping file like icbm_adult.map.gz
            dsistudiotemplate = self.inputs.fibgz_map.split(".")[-3]
            new_mapping_file = merged_fibgz_file + f".{dsistudiotemplate}.map.gz"
            shutil.copyfile(self.inputs.fibgz_map, new_mapping_file)
            self._results["fibgz_map"] = new_mapping_file

        return runtime


def get_dsi_studio_ODF_geometry(odf_key):
    mat_path = pkgr("qsirecon", "data/odfs.mat")
    m = loadmat(mat_path)
    odf_vertices = m[odf_key + "_vertices"].T
    odf_faces = m[odf_key + "_faces"].T
    return odf_vertices, odf_faces


def popen_run(arg_list):
    cmd = subprocess.Popen(arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    LOGGER.info(out)
    LOGGER.info(err)


def combine_gqi_and_csd_fib_files(path_gqi_fib: str, path_fod_fib: str, merged_fib: str):
    """
    Combine the GQI and CSD .fib-files such that the new CSD fib file contains ODF information
    from the old CSD file and DTI maps from the GQI file.

    Args:
      path_gqi_file: Full path to the GQI file
      path_csd_file: Full path to the csd file. This will be overwritten with the updated csd file.
    """

    gqi_data = fast_load_fibgz(path_gqi_fib)
    fod_data = fast_load_fibgz(path_fod_fib)
    merged_data = gqi_data.copy()

    for gqi_key in gqi_data:
        if (
            re.match(r"odf\d+", gqi_key)
            or re.match(r"fa\d+", gqi_key)
            or re.match(r"index\d+", gqi_key)
        ):
            LOGGER.info(f"Deleting {gqi_key} from GQI data")
            del merged_data[gqi_key]
    for fod_key in fod_data:
        if (
            re.match(r"odf\d+", fod_key)
            or re.match(r"fa\d+", fod_key)
            or re.match(r"index\d+", fod_key)
        ):
            LOGGER.info(f"Copying {fod_key} from FOD data")
            merged_data[fod_key] = fod_data[fod_key]

    savemat(merged_fib, merged_data, format="4", appendmat=False)


def amplitudes_to_fibgz(
    amplitudes_img, odf_dirs, odf_faces, output_file, mask_img, num_fibers=5, unit_odf=False
):
    """Convert a NiftiImage of ODF amplitudes to a DSI Studio fib file.

    Parameters:
    ===========

    amplitudes_img: nb.Nifti1Image
        4d NIfTI image that contains amplitudes for the ODFs
    odf_dirs: np.ndarray
        N x 3 array containing the directions corresponding to the
        amplitudes in ``amplitudes_img``. The values in
        ``amplitudes_img.get_fdata()[..., i]`` are for the
        direction in ``odf_dirs[i]``.
    odf_faces: np.ndarray
        triangles connecting the vertices in ``odf_dirs``
    output_file: str
        Path where the output fib file will be written.
    mask_img: nb.Nifti1Image
        3d Image that is nonzero where voxels contain brain.
    num_fibers: int
        The maximum number of fibers/fixels stored in each voxel.

    Returns:
    ========

    None


    """
    num_dirs, _ = odf_dirs.shape
    hemisphere = num_dirs // 2
    x, y, z = odf_dirs[:hemisphere].T
    hs = HemiSphere(x=x, y=y, z=z)

    if not np.allclose(mask_img.affine, amplitudes_img.affine):
        raise ValueError("Differing orientation between mask and amplitudes")
    if not mask_img.shape == amplitudes_img.shape[:3]:
        raise ValueError("Differing grid between mask and amplitudes")

    # Get the flat mask
    ampl_data = amplitudes_img.get_fdata()
    flat_mask = mask_img.get_fdata().flatten(order="F") > 0
    odf_array = ampl_data.reshape(-1, ampl_data.shape[3], order="F")
    del ampl_data
    masked_odfs = odf_array[flat_mask, :]
    z0 = np.nanmax(masked_odfs)
    masked_odfs = masked_odfs / z0
    masked_odfs[masked_odfs < 0] = 0
    masked_odfs = np.nan_to_num(masked_odfs).astype(float)

    if unit_odf:
        sums = masked_odfs.sum(1)
        sums[sums == 0] = 1
        masked_odfs = np.masked_odfs / sums[:, np.newaxis]

    n_odfs = masked_odfs.shape[0]
    peak_indices = np.zeros((n_odfs, num_fibers))
    peak_vals = np.zeros((n_odfs, num_fibers))

    dsi_mat = {}
    # Create matfile that can be read by dsi Studio
    dsi_mat["dimension"] = np.array(amplitudes_img.shape[:3])
    dsi_mat["voxel_size"] = np.array(amplitudes_img.header.get_zooms()[:3])
    n_voxels = int(np.prod(dsi_mat["dimension"]))
    LOGGER.info("Detecting Peaks")
    for odfnum in range(n_odfs):
        dirs, vals, indices = peak_directions(masked_odfs[odfnum], hs)
        for dirnum, (val, idx) in enumerate(zip(vals, indices)):
            if dirnum == num_fibers:
                break
            peak_indices[odfnum, dirnum] = idx
            peak_vals[odfnum, dirnum] = val

    # ensure that fa0 > 0 for all odf values
    peak_vals[np.abs(peak_vals[:, 0]) < MIN_NONZERO, 0] = MIN_NONZERO
    for nfib in range(num_fibers):
        # fill in the "fa" values
        fa_n = np.zeros(n_voxels)
        fa_n[flat_mask] = peak_vals[:, nfib]
        dsi_mat["fa%d" % nfib] = fa_n.astype("float32")

        # Fill in the index values
        index_n = np.zeros(n_voxels)
        index_n[flat_mask] = peak_indices[:, nfib]
        dsi_mat["index%d" % nfib] = index_n.astype("int16")

    # Add in the ODFs
    num_odf_matrices = n_odfs // ODF_COLS
    split_indices = (np.arange(num_odf_matrices) + 1) * ODF_COLS
    odf_splits = np.array_split(masked_odfs, split_indices, axis=0)
    for splitnum, odfs in enumerate(odf_splits):
        dsi_mat["odf%d" % splitnum] = odfs.T.astype("float32")

    dsi_mat["odf_vertices"] = odf_dirs.T
    dsi_mat["odf_faces"] = odf_faces.T
    dsi_mat["z0"] = np.array([z0])
    savemat(output_file, dsi_mat, format="4", appendmat=False)


def amico_directions_to_fibgz(
    directions_img,
    od_img,
    icvf_img,
    modulated_od_img,
    modulated_icvf_img,
    isovf_img,
    odf_dirs,
    odf_faces,
    output_file,
    mask_img,
):
    """Convert a NiftiImage of ODF amplitudes to a DSI Studio fib file.

    Parameters:
    ===========

    directions_img: nb.Nifti1Image (I x J x K x 3)
        peak directions image from NODDI fit
    od_img: nb.Nifti1Image
        orientation dispersion image
    icvf_img: nb.Nifti1Image
        icvf image
    isovf_img: nb.Nifti1Image
        isovf image
    odf_dirs: np.ndarray
        N x 3 array containing the directions corresponding to the
        amplitudes in ``amplitudes_img``. The values in
        ``amplitudes_img.get_fdata()[..., i]`` are for the
        direction in ``odf_dirs[i]``.
    odf_faces: np.ndarray
        triangles connecting the vertices in ``odf_dirs``
    output_file: str
        Path where the output fib file will be written.
    mask_img: nb.Nifti1Image
        3d Image that is nonzero where voxels contain brain.
    num_fibers: int
        The maximum number of fibers/fixels stored in each voxel.

    Returns:
    ========

    None


    """
    num_dirs, _ = odf_dirs.shape
    hemisphere = num_dirs // 2
    x, y, z = odf_dirs[:hemisphere].T
    hs = HemiSphere(x=x, y=y, z=z)

    if not np.allclose(mask_img.affine, directions_img.affine):
        raise ValueError("Differing orientation between mask and directions")
    if not mask_img.shape == directions_img.shape[:3]:
        raise ValueError("Differing grid between mask and amplitudes")

    # Get the flat mask
    directions_data = directions_img.get_fdata()
    flat_mask = mask_img.get_fdata().flatten(order="F") > 0
    n_odfs = flat_mask.sum()
    directions_array = directions_data.reshape(-1, directions_data.shape[3], order="F")
    # directions_array[:, 1] = -directions_array[:, 1]
    directions_array[:, 0] = -directions_array[:, 0]
    masked_dirs = np.nan_to_num(directions_array[flat_mask, :])
    isovf_vec = isovf_img.get_fdata().flatten(order="F")
    icvf_vec = icvf_img.get_fdata().flatten(order="F")
    od_vec = od_img.get_fdata().flatten(order="F")
    mod_icvf_vec = modulated_icvf_img.get_fdata().flatten(order="F")
    mod_od_vec = modulated_od_img.get_fdata().flatten(order="F")

    # z0 = np.nanmax(isovf_vec)
    peak_indices = np.zeros(n_odfs)

    # Create matfile that can be read by dsi Studio
    dsi_mat = {}
    dsi_mat["dimension"] = np.array(directions_img.shape[:3])
    dsi_mat["voxel_size"] = np.array(directions_img.header.get_zooms()[:3])
    n_voxels = int(np.prod(dsi_mat["dimension"]))
    LOGGER.info("Detecting Peaks")
    for odfnum in range(n_odfs):
        peak_indices[odfnum] = hs.find_closest(masked_dirs[odfnum])

    # fill in the "dir" values
    dir0 = np.zeros(n_voxels)
    dir0[flat_mask] = peak_indices
    dsi_mat["index0"] = dir0.astype("int16")
    dsi_mat["fa0"] = icvf_vec
    dsi_mat["icvf0"] = icvf_vec
    dsi_mat["isovf0"] = isovf_vec
    dsi_mat["od0"] = od_vec
    dsi_mat["mod_icvf0"] = mod_icvf_vec
    dsi_mat["mod_od0"] = mod_od_vec
    dsi_mat["odf_vertices"] = odf_dirs.T
    dsi_mat["odf_faces"] = odf_faces.T
    savemat(output_file, dsi_mat, format="4", appendmat=False)


def amplitudes_to_sh_mif(amplitudes_img, odf_dirs, output_file, working_dir):
    """Convert an image of ODF amplitudes to a MRtrix sh mif file.

    Parameters:
    ============

    amplitudes_img: nb.Nifti1Image
        4d NIfTI image that contains amplitudes for the ODFs
    odf_dirs: np.ndarray
        2*N x 3 array containing the directions corresponding to the
        amplitudes in ``amplitudes_img``. The values in
        ``amplitudes_img.get_fdata()[..., i]`` are for the
        direction in ``odf_dirs[i]``. Here the second half of the
        directions are the opposite of the fist and therefore have the
        same amplitudes.
    output_file: str
        Path where the output ``.mif`` file will be written.
    working_dir: str
        Path where temp files will be written to

    Returns:
    ========

    None

    """
    temp_nii = op.join(working_dir, "odf_values.nii")
    amplitudes_img.to_filename(temp_nii)

    num_dirs, _ = odf_dirs.shape
    hemisphere = num_dirs // 2
    x, y, z = odf_dirs[:hemisphere].T
    _, theta, phi = cart2sphere(-x, -y, z)
    dirs_txt = op.join(working_dir, "ras+directions.txt")
    np.savetxt(dirs_txt, np.column_stack([phi, theta]))

    popen_run(
        ["amp2sh", "-quiet", "-force", "-directions", dirs_txt, "odf_values.nii", output_file]
    )
    os.remove(temp_nii)
    os.remove(dirs_txt)


def mif2amps(sh_mif_file, working_dir, dsi_studio_odf="odf8"):
    """Convert a MRTrix SH mif file to a NiBabel amplitudes image.

    Parameters:
    ===========

    sh_mif_file : str
        path to the mif file with SH coefficients

    """
    verts, _ = get_dsi_studio_ODF_geometry(dsi_studio_odf)
    num_dirs, _ = verts.shape
    hemisphere = num_dirs // 2
    directions = verts[:hemisphere]
    x, y, z = directions.T
    _, theta, phi = cart2sphere(x, y, -z)
    dirs_txt = op.join(working_dir, "directions.txt")
    np.savetxt(dirs_txt, np.column_stack([phi, theta]))

    odf_amplitudes_nii = op.join(working_dir, "amplitudes.nii")
    popen_run(["sh2amp", "-quiet", "-nonnegative", sh_mif_file, dirs_txt, odf_amplitudes_nii])

    if not op.exists(odf_amplitudes_nii):
        raise FileNotFoundError("Unable to create %s", odf_amplitudes_nii)
    amplitudes_img = nb.load(odf_amplitudes_nii)
    return amplitudes_img, directions


def fast_load_fibgz(fib_file):
    """Load a potentially gzipped fibgz file more quickly than using built-in gzip."""
    # Try to load a non-zipped file
    if not fib_file.endswith("gz"):
        return loadmat(fib_file)

    # Load a zipped file quickly if possible
    def find_zcat():
        def is_exe(fpath):
            return os.path.exists(fpath) and os.access(fpath, os.X_OK)

        for program in ["gzcat", "zcat"]:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return program
        return None

    # Check if a zcat is available on this system:
    zcatter = find_zcat()
    if zcatter is not None:
        p = subprocess.Popen([zcatter, fib_file], stdout=subprocess.PIPE)
        fh = StringIO(p.communicate()[0])
        assert p.returncode == 0
        return loadmat(fh)

    with gzip.open(fib_file, "r") as f:
        LOGGER.info("Loading with python gzip. To load faster install zcat or gzcat.")
        return loadmat(f)


def fib2amps(fib_file, ref_image, subtract_iso=True):
    fibmat = fast_load_fibgz(fib_file)
    dims = tuple(fibmat["dimension"].squeeze().astype(int))
    directions = fibmat["odf_vertices"].T

    odf_vars = [k for k in fibmat.keys() if re.match("odf\\d+", k)]
    valid_odfs = []
    flat_mask = fibmat["fa0"].squeeze().ravel(order="F") > 0
    n_voxels = np.prod(dims)
    if odf_vars:
        for n in range(len(odf_vars)):
            varname = "odf%d" % n
            odfs = fibmat[varname]
            odf_sum = odfs.sum(0)
            odf_sum_mask = odf_sum > 0
            valid_odfs.append(odfs[:, odf_sum_mask].T)
        odf_array = np.row_stack(valid_odfs)
        if subtract_iso:
            odf_array = odf_array - odf_array.min(0)
    else:
        odf_array = peaks_to_odfs(fibmat)

    # Convert each column to a 3d file, then concatenate them
    odfs_3d = []
    for odf_vals in odf_array.T:
        new_data = np.zeros(n_voxels, dtype="float32")
        new_data[flat_mask] = odf_vals
        odfs_3d.append(new_data.reshape(dims, order="F"))

    real_img = nb.load(ref_image)
    odf4d = np.stack(odfs_3d, -1)
    odf4d_img = nb.Nifti1Image(odf4d, real_img.affine, real_img.header)

    return odf4d_img, directions


def peaks_to_odfs(fibdict):
    """If no ODF data is available, create fake ODFs that will behave properly."""
    index_vars = [key for key in fibdict if key.startswith("index")]
    num_indexes = len(index_vars)
    flat_mask = fibdict["fa0"].squeeze().ravel(order="F") > 0
    num_directions = fibdict["odf_vertices"].shape[1]
    num_odfs = flat_mask.sum()
    odfs = np.zeros((num_odfs, num_directions), dtype="float32")
    row_indices = np.arange(num_odfs)
    for peak_num in range(num_indexes):
        fa_values = fibdict["fa%d" % peak_num].squeeze().ravel(order="F")[flat_mask]
        peak_indices = fibdict["index%d" % peak_num].squeeze().ravel(order="F")[flat_mask]
        odfs[row_indices, peak_indices] = fa_values
    return odfs
