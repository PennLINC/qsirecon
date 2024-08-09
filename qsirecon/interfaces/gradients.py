"""Handle merging and spliting of DSI files."""

import logging
import os

import nibabel as nb
import numpy as np
from nilearn import image as nim
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from sklearn.metrics import r2_score

LOGGER = logging.getLogger("nipype.interface")
tensor_index = {"xx": (0, 0), "xy": (0, 1), "xz": (0, 2), "yy": (1, 1), "yz": (1, 2), "zz": (2, 2)}


class RemoveDuplicatesInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    local_bvec_file = File(exists=True)
    distance_cutoff = traits.Float(5.0, usedefault=True)
    expected_directions = traits.Int()


class RemoveDuplicatesOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    local_bvec_file = File(exists=True)


class RemoveDuplicates(SimpleInterface):
    input_spec = RemoveDuplicatesInputSpec
    output_spec = RemoveDuplicatesOutputSpec

    def _run_interface(self, runtime):
        bvecs = np.loadtxt(self.inputs.bvec_file).T
        bvals = np.loadtxt(self.inputs.bval_file).squeeze()
        orig_bvals = bvals.copy()
        bvals = np.sqrt(bvals - bvals.min())
        bvals = bvals / bvals.max() * 100
        original_image = nb.load(self.inputs.dwi_file)
        cutoff = self.inputs.distance_cutoff

        scaled_bvecs = bvals[:, np.newaxis] * bvecs
        ok_vecs = []
        seen_vecs = []

        def is_unique_sample(vec):
            if len(seen_vecs) == 0:
                return True
            vec_array = np.row_stack(seen_vecs)
            distances = np.linalg.norm(vec_array - vec, axis=1)
            distances_flip = np.linalg.norm(vec_array + vec, axis=1)
            return np.all(distances > cutoff) and np.all(distances_flip > cutoff)

        for vec_num, vec in enumerate(scaled_bvecs):
            magnitude = np.linalg.norm(vec)
            # Is it a b0?
            if magnitude < cutoff:
                ok_vecs.append(vec_num)
            else:
                if is_unique_sample(vec):
                    ok_vecs.append(vec_num)
                    seen_vecs.append(vec)

        # If an expected number of directions was specified, check that it's there
        expected = self.inputs.expected_directions
        if isdefined(expected):
            if not len(seen_vecs) == expected:
                raise Exception(
                    "Expected %d unique samples but found %d", expected, len(seen_vecs)
                )

        # Are all the directions unique?
        if len(ok_vecs) == len(bvals):
            self._results["dwi_file"] = self.inputs.dwi_file
            self._results["bval_file"] = self.inputs.bval_file
            self._results["bvec_file"] = self.inputs.bvec_file
            self._results["local_bvec_file"] = self.inputs.local_bvec_file

            return runtime

        # Extract the unique samples
        output_bval = fname_presuffix(self.inputs.bval_file, newpath=runtime.cwd, suffix="_unique")
        output_bvec = fname_presuffix(self.inputs.bvec_file, newpath=runtime.cwd, suffix="_unique")
        output_nii = fname_presuffix(self.inputs.dwi_file, newpath=runtime.cwd, suffix="_unique")
        unique_indices = np.array(ok_vecs)
        unique_bvals = orig_bvals[unique_indices]
        np.savetxt(output_bval, unique_bvals, fmt="%d", newline=" ")
        unique_bvecs = bvecs[unique_indices]
        np.savetxt(output_bvec, unique_bvecs.T, fmt="%.8f")
        unique_data = original_image.get_fdata()[..., unique_indices]
        nb.Nifti1Image(unique_data, original_image.affine, original_image.header).to_filename(
            output_nii
        )
        self._results["bval_file"] = output_bval
        self._results["bvec_file"] = output_bvec
        self._results["dwi_file"] = output_nii
        # TODO: support local bvecs
        return runtime


class SliceQCInputSpec(BaseInterfaceInputSpec):
    uncorrected_dwi_files = InputMultiObject(File(exists=True), desc="uncorrected dwi files")
    ideal_image_files = InputMultiObject(File(exists=True), desc="model-based images")
    mask_image = File(exists=True, desc="brain mask")
    impute_slice_threshold = traits.Float(0.0, desc="threshold for using imputed data in a slice")
    min_slice_size_percentile = traits.CFloat(
        10.0,
        usedefault=True,
        desc="slices bigger than " "this percentile are candidates for imputation.",
    )


class SliceQCOutputSpec(TraitedSpec):
    imputed_images = OutputMultiObject(File(exists=True), desc="dwi files with imputed slices")
    slice_stats = File(exists=True, desc="npy file with the slice-by-TR error matrix")


class SliceQC(SimpleInterface):
    input_spec = SliceQCInputSpec
    output_spec = SliceQCOutputSpec

    def _run_interface(self, runtime):
        ideal_image_files = self.inputs.ideal_image_files
        uncorrected_image_files = self.inputs.uncorrected_dwi_files

        self._results["imputed_images"] = self.inputs.uncorrected_dwi_files
        output_npz = os.path.join(runtime.cwd, "slice_stats.npz")
        mask_img = nb.load(self.inputs.mask_image)
        mask = mask_img.get_fdata() > 0
        masked_slices = (mask * np.arange(mask_img.shape[2])[np.newaxis, np.newaxis, :]).astype(
            int
        )
        slice_nums, slice_counts = np.unique(masked_slices[mask], return_counts=True)
        min_size = np.percentile(slice_counts, self.inputs.min_slice_size_percentile)
        too_small = slice_nums[slice_counts < min_size]
        for small_slice in too_small:
            masked_slices[masked_slices == small_slice] = 0
        valid_slices = slice_nums[slice_counts > min_size]
        valid_slices = valid_slices[valid_slices > 0]
        slice_scores = []
        wb_xcorrs = []
        wb_r2s = []
        # If impute slice threshold==0 or hmc_model=="none"
        if isdefined(ideal_image_files):
            for ideal_image, input_image in zip(ideal_image_files, uncorrected_image_files):
                slices, wb_xcorr, wb_r2 = _score_slices(
                    ideal_image, input_image, masked_slices, valid_slices
                )
                slice_scores.append(slices)
                wb_xcorrs.append(wb_xcorr)
                wb_r2s.append(wb_r2)
        else:
            num_trs = len(uncorrected_image_files)
            num_slices = mask_img.shape[2]
            wb_xcorrs = np.zeros(num_trs)
            wb_r2s = np.zeros(num_trs)
            slice_scores = np.zeros((num_slices, num_trs))

        np.savez(
            output_npz,
            slice_scores=slice_scores,
            wb_r2s=np.array(wb_r2s),
            wb_xcorrs=np.array(wb_xcorrs),
            valid_slices=valid_slices,
            masked_slices=masked_slices,
            slice_nums=slice_nums,
            slice_counts=slice_counts,
        )
        self._results["slice_stats"] = output_npz
        return runtime


def _score_slices(ideal_image, input_image, masked_slices, valid_slices):
    """Compute similarity metrics on a pair of images."""

    def crosscor(vec1, vec2):
        v1bar = vec1 - vec1.mean()
        v2bar = vec2 - vec2.mean()
        return np.inner(v1bar, v2bar) ** 2 / (np.inner(v1bar, v1bar) * np.inner(v2bar, v2bar))

    slice_scores = np.zeros(valid_slices.shape)
    ideal_data = nb.load(ideal_image).get_fdata()
    input_data = nb.load(input_image).get_fdata()
    for nslice, slicenum in enumerate(valid_slices):
        slice_mask = masked_slices == slicenum
        ideal_slice = ideal_data[slice_mask]
        data_slice = input_data[slice_mask]
        slice_scores[nslice] = crosscor(ideal_slice, data_slice)

    global_mask = masked_slices > 0
    wb_ideal = ideal_data[global_mask]
    wb_input = input_data[global_mask]
    global_xcorr = crosscor(wb_input, wb_ideal)
    global_r2 = r2_score(wb_input, wb_ideal)
    return slice_scores, global_xcorr, global_r2


class ExtractB0sInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List()
    bval_file = File(exists=True)
    b0_threshold = traits.Int(50, usedefault=True)
    dwi_series = File(exists=True, mandatory=True)


class ExtractB0sOutputSpec(TraitedSpec):
    b0_series = File(exists=True)
    b0_average = File(exists=True)


class ExtractB0s(SimpleInterface):
    """Extract a b0 series and a mean b0 from a dwi series."""

    input_spec = ExtractB0sInputSpec
    output_spec = ExtractB0sOutputSpec

    def _run_interface(self, runtime):
        output_fname = fname_presuffix(
            self.inputs.dwi_series, suffix="_b0_series", use_ext=True, newpath=runtime.cwd
        )
        output_mean_fname = fname_presuffix(
            output_fname, suffix="_mean", use_ext=True, newpath=runtime.cwd
        )
        if isdefined(self.inputs.b0_indices):
            indices = np.array(self.inputs.b0_indices).astype(int)
        elif isdefined(self.inputs.bval_file):
            bvals = np.loadtxt(self.inputs.bval_file)
            indices = np.flatnonzero(bvals < self.inputs.b0_threshold)
            if indices.size == 0:
                raise ValueError("No b<%d images found" % self.inputs.b0_threshold)
        else:
            raise ValueError("No gradient information available")
        new_data = nim.index_img(self.inputs.dwi_series, indices)
        new_data.to_filename(output_fname)
        self._results["b0_series"] = output_fname
        if new_data.ndim == 3:
            self._results["b0_average"] = output_fname
        else:
            mean_image = nim.math_img("img.mean(3)", img=new_data)
            mean_image.to_filename(output_mean_fname)
            self._results["b0_average"] = output_mean_fname

        return runtime


def concatenate_bvals(bval_list, out_file):
    """Create an FSL-style bvals file from split bval files."""
    collected_vals = []
    for bval_file in bval_list:
        collected_vals.append(np.loadtxt(bval_file, ndmin=1))
    final_bvals = np.concatenate(collected_vals).squeeze()
    if out_file is not None:
        np.savetxt(out_file, final_bvals, fmt=str("%i"))
    return final_bvals


def concatenate_bvecs(input_files):
    """Create Dipy-style gradient array (3-columns) from bvec files."""
    if len(input_files) == 1:
        stacked = np.loadtxt(input_files[0])
    else:
        collected_vecs = []
        for bvec_file in input_files:
            collected_vecs.append(np.loadtxt(bvec_file).astype(float))
            stacked = np.row_stack(collected_vecs)
    if not stacked.shape[1] == 3:
        stacked = stacked.T
    return stacked


def write_concatenated_fsl_gradients(bval_files, bvec_files, out_prefix):
    bvec_file = out_prefix + ".bvec"
    bval_file = out_prefix + ".bval"
    stacked_bvecs = concatenate_bvecs(bvec_files)
    np.savetxt(bvec_file, stacked_bvecs.T, fmt="%.8f", delimiter=" ")
    concatenate_bvals(bval_files, bval_file)
    return bval_file, bvec_file


def get_vector_nii(data, affine, header):
    hdr = header.copy()
    hdr.set_data_dtype(np.dtype("<f4"))
    hdr.set_intent("vector", (), "")
    return nb.Nifti1Image(data[:, :, :, np.newaxis, :].astype(np.dtype("<f4")), affine, hdr)
