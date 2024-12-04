"""Handle merging and spliting of DSI files."""

import logging

import nibabel as nb
import numpy as np
from nilearn import image as nim
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    InputMultiObject,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger("nipype.interface")


class _GradientSelectInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    bval_distance_cutoff = traits.Float(5.0, usedefault=True)
    expected_n_input_shells = traits.Int()
    expected_n_output_shells = traits.Int()
    requested_shell_bvals = InputMultiObject(traits.CInt(), mandatory=True)


class _GradientSelectOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    local_bvec_file = File(exists=True)


class GradientSelect(SimpleInterface):
    input_spec = _GradientSelectInputSpec
    output_spec = _GradientSelectOutputSpec

    def _run_interface(self, runtime):
        """Find shells in the input data and select"""

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.d
        max_distance = self.inputs.bval_distance_cutoff
        bvals = np.loadtxt(self.inputs.bval_file)
        agg_cluster = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=2 * max_distance,
            linkage="complete",
        ).fit(bvals.reshape(-1, 1))

        # Check that the correct number of input shells are detected
        if isdefined(self.inputs.expected_n_input_shells):
            if not self.inputs.expected_n_input_shells == agg_cluster.n_clusters_:
                howmuch, newthresh = (
                    ("too many", "higher")
                    if agg_cluster.n_clusters_ > self.inputs.expected_n_input_shells
                    else ("too few", "lower")
                )
                raise Exception(
                    f"Expected to find {self.inputs.expected_n_input_shells} shells in "
                    f"the input data. Instead we found {agg_cluster.n_clusters_}. Having "
                    f"{howmuch} shells detected means you may need to adjust the"
                    f"bval_distance_cutoff parameter to be {newthresh}."
                )

        # Ensure that at lease 1 b>0 and one b=0 shell are selected
        select_shell_bs = self.inputs.requested_shell_bvals
        if len(select_shell_bs) < 2:
            if select_shell_bs[0] < max_distance:
                LOGGER.critical("No effectively b>0 shells are selected.")
            else:
                LOGGER.warning("Adding a b=0 shell to the selection.")
                select_shell_bs.append(0)

        # Make sure the shells are unique
        select_shell_bs = np.array(sorted(select_shell_bs))
        shell_distances =



        return runtime


def _select_shell_by_bval(desired_bval, distance_max, bvals, bval_assignments=None):
    """Find indices where bvals are within distance_max of desired_bval.

    If bval_assignments is defined,
    """


class _RemoveDuplicatesInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    local_bvec_file = File(exists=True)
    distance_cutoff = traits.Float(5.0, usedefault=True)
    expected_directions = traits.Int()


class _RemoveDuplicatesOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    local_bvec_file = File(exists=True)


class RemoveDuplicates(SimpleInterface):
    input_spec = _RemoveDuplicatesInputSpec
    output_spec = _RemoveDuplicatesOutputSpec

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
