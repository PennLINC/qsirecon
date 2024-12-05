"""Handle merging and spliting of DSI files."""

import logging

import numpy as np
from nilearn import image as nim
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
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
    b_file = File(exists=True)
    btable_file = File(exists=True)
    bval_distance_cutoff = traits.Float(5.0, usedefault=True)
    expected_n_input_shells = traits.Int()
    requested_shells = InputMultiObject(
        traits.Either(traits.CInt(), traits.Enum("highest", "lowest")), mandatory=True
    )


class _GradientSelectOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    btable_file = File(exists=True)
    local_bvec_file = File(exists=True)


class GradientSelect(SimpleInterface):
    input_spec = _GradientSelectInputSpec
    output_spec = _GradientSelectOutputSpec

    def _run_interface(self, runtime):
        """Find shells in the input data and select"""
        bvals = np.loadtxt(self.inputs.bval_file)

        selected_indices = _select_gradients(
            requested_gradients=self.inputs.requested_shells,
            max_distance=self.inputs.bval_distance_cutoff,
            original_bvals=bvals,
            expected_n_input_shells=self.inputs.expected_n_input_shells or None,
        )

        # Make a new dwi and associated gradient files
        new_bval, new_bvec, new_b, new_btable, new_nifti = subset_dwi(
            original_bval=self.inputs.bval_file,
            original_bvec=self.inputs.bvec_file,
            original_b=self.inputs.b_file or None,
            original_btable=self.inputs.btable_file or None,
            original_nifti=self.inputs.dwi_file,
            indices=selected_indices,
            newdir=runtime.cwd,
            suffix="_selected",
        )
        self._results["bval_file"] = new_bval
        self._results["bvec_file"] = new_bvec
        self._results["dwi_file"] = new_nifti
        if new_b:
            self._results["b_file"] = new_b
        if new_btable:
            self._results["btable_file"] = new_btable

        return runtime


def _select_gradients(
    requested_gradients,
    max_distance,
    original_bvals,
    expected_n_input_shells=None,
):
    """Find indices where bvals are within distance_max of desired_bval."""
    from sklearn.metrics import pairwise_distances

    # Get the bvals clustered into assigned shells. Also in a dataframe
    bval_df = _find_shells(original_bvals, max_distance)

    # Check that the correct number of input shells are detected
    if expected_n_input_shells is not None:
        n_found_shells = len(bval_df["assigned_shell"].unique())
        if not expected_n_input_shells == n_found_shells:
            howmuch, newthresh = (
                ("too many", "higher")
                if n_found_shells > expected_n_input_shells
                else ("too few", "lower")
            )
            raise Exception(
                f"Expected to find {expected_n_input_shells} shells in "
                f"the input data. Instead we found {n_found_shells}. Having "
                f"{howmuch} shells detected means you may need to adjust the"
                f"bval_distance_cutoff parameter to be {newthresh}."
            )

    # Ensure that at lease 1 b>0 and one b=0 shell are selected
    select_shell_bs = _parse_shell_selection(requested_gradients, bval_df, max_distance)

    # Make sure the shells are unique (no overlap when accounting for allowed distances)
    shell_distances = pairwise_distances(select_shell_bs.reshape(-1, 1))
    if np.any(shell_distances[np.triu_indices_from(shell_distances, k=1)] < max_distance):
        raise Exception(
            "Shells and bval_distance_cutoff have overlap. Choose a lower "
            "bval_distance_cutoff or more separated shells."
        )

    selected_indices = np.flatnonzero(bval_df["assigned_shell"].isin(select_shell_bs))

    return selected_indices


def _parse_shell_selection(requested_bvals, bval_df, max_distance):
    """Turn bval requests into numbers. Add a 0 if none were originally included."""
    numeric_bvals = []
    for requested_bval in requested_bvals:
        if requested_bval == "highest":
            highest_shell = bval_df["assigned_shell"].max()
            LOGGER.info(f"Selecting b={highest_shell} as the highest shell")
            numeric_bvals.append(highest_shell)
        elif requested_bval == "lowest":
            lowest_shell = bval_df["assigned_shell"][
                bval_df["assigned_shell"] > max_distance
            ].min()
            LOGGER.info(f"Selecting b={lowest_shell} as the lowest b>0 shell")
            numeric_bvals.append(lowest_shell)
        else:
            # Find the closest detected shell
            ok_shells = bval_df["assigned_shell"][
                np.logical_and(
                    bval_df["assigned_shell"] >= requested_bval - max_distance,
                    bval_df["assigned_shell"] <= requested_bval + max_distance,
                )
            ]
            if len(ok_shells.unique()) > 1:
                raise Exception(
                    f"Unable to unambiguously select b={requested_bval}."
                    f"instead select from {bval_df.assigned_shells.unique().tolist()}"
                )

            numeric_bvals.append(ok_shells.iloc[0])

    # Make sure there is a 0 in the list
    if len(numeric_bvals) < 2:
        if numeric_bvals[0] < max_distance:
            LOGGER.critical("No effectively b>0 shells are selected.")
        else:
            LOGGER.warning("Adding a b=0 shell to the selection.")
            numeric_bvals.append(0)

    return np.array(sorted(numeric_bvals))


def _find_shells(bvals, max_distance):

    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    X = bvals.reshape(-1, 1)
    agg_cluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=2 * max_distance,
        linkage="complete",
    ).fit(X)
    shells = agg_cluster.labels_

    score = silhouette_score(X, shells)
    if score < 0.8:
        print("Silhouette score is low. Is this is a DSI scheme?")

    # Do the same check as mrtrix
    max_shells = np.sqrt(np.sum(bvals > max_distance))
    if agg_cluster.n_clusters_ > max_shells:
        raise Exception("Too many possible shells detected.")

    bval_df = pd.DataFrame({"bvalue": bvals, "assignment": shells})
    shell_df = bval_df.groupby("assignment", as_index=False).agg({"bvalue": "median"})
    bval_df["assigned_shell"] = bval_df["assignment"].replace(
        shell_df["assignment"].tolist(), shell_df["bvalue"].tolist()
    )
    bval_df["shell_num"] = bval_df["assigned_shell"].rank(method="dense")
    bval_df.drop(columns=["assignment"], inplace=True)

    return bval_df


class _RemoveDuplicatesInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    b_file = File(exists=True)
    btable_file = File(exists=True)
    local_bvec_file = File(exists=True)
    distance_cutoff = traits.Float(5.0, usedefault=True)
    expected_directions = traits.Int()


class _RemoveDuplicatesOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    b_file = File(exists=True)
    btable_file = File(exists=True)
    local_bvec_file = File(exists=True)


class RemoveDuplicates(SimpleInterface):
    input_spec = _RemoveDuplicatesInputSpec
    output_spec = _RemoveDuplicatesOutputSpec

    def _run_interface(self, runtime):
        bvecs = np.loadtxt(self.inputs.bvec_file).T
        bvals = np.loadtxt(self.inputs.bval_file).squeeze()
        bvals = np.sqrt(bvals - bvals.min())
        bvals = bvals / bvals.max() * 100
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

        # Extract the unique samples
        new_bval, new_bvec, new_b, new_btable, new_nifti = subset_dwi(
            original_bval=self.inputs.bval_file,
            original_bvec=self.inputs.bvec_file,
            original_b=self.inputs.b_file or None,
            original_btable=self.inputs.btable_file or None,
            original_nifti=self.inputs.dwi_file,
            indices=np.array(ok_vecs),
            newdir=runtime.cwd,
            suffix="_unique",
        )
        self._results["bval_file"] = new_bval
        self._results["bvec_file"] = new_bvec
        self._results["dwi_file"] = new_nifti
        if new_b:
            self._results["b_file"] = new_b
        if new_btable:
            self._results["btable_file"] = new_btable

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


def subset_dwi(
    original_bval,
    original_bvec,
    original_b,
    original_btable,
    original_nifti,
    indices,
    newdir,
    suffix="_bsel",
):
    """Create a subset of a dwi based on a set of indices."""
    bvals = np.loadtxt(original_bval)
    if np.all(indices == np.arange(len(bvals))):
        return original_bval, original_bvec, original_b, original_btable, original_nifti

    # Subset and write the bval
    new_bval = fname_presuffix(original_bval, newpath=newdir, suffix=suffix)
    subsetted_bval_data = np.loadtxt(original_bval).squeeze()[indices]
    np.savetxt(new_bval, subsetted_bval_data, fmt="%d", newline=" ")

    # Subset and write the bvec
    new_bvec = fname_presuffix(original_bvec, newpath=newdir, suffix=suffix)
    selected_bvecs = np.loadtxt(original_bvec)[:, indices]
    np.savetxt(new_bvec, selected_bvecs, fmt="%.8f")

    # Subset and write the dwi nifti
    new_nifti = fname_presuffix(original_nifti, newpath=newdir, suffix=suffix)
    nim.index_img(original_nifti, indices).to_filename(new_nifti)

    # If there is a b file, subset and write it
    if original_b is not None:
        new_b = fname_presuffix(original_b, newpath=newdir, suffix=suffix)
        _select_lines(original_b, new_b, indices)

    # If there is a dsi studio btable file, subset and write it
    if original_btable is not None:
        new_btable = fname_presuffix(original_btable, newpath=newdir, suffix=suffix)
        _select_lines(original_btable, new_btable, indices)
    return new_bval, new_bvec, new_b, new_btable, new_nifti


def _select_lines(in_file, out_file, indices):

    with open(in_file, "r") as in_f:
        in_lines = in_f.readlines()
        new_lines = [in_lines[lineno] for lineno in indices]

    with open(out_file, "w") as out_f:
        out_f.writelines(new_lines)
