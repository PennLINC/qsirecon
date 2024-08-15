# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""

import gzip
import os
import os.path as op
import re
from json import loads
from shutil import copyfileobj, copytree

from bids.layout import Config, parse_file_entities
from nipype import logging
from nipype.interfaces.base import File, InputMultiObject, isdefined, traits
from nipype.utils.filemanip import copyfile
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink
from niworkflows.interfaces.bids import _DerivativesDataSinkInputSpec

from qsirecon import config
from qsirecon.data import load as load_data

LOGGER = logging.getLogger("nipype.interface")
BIDS_NAME = re.compile(
    "^(.*\/)?(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?"
    "(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?"
    "(_(?P<space_id>space-[a-zA-Z0-9]+))?"
    "(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?"
)

# NOTE: Modified for QSIRecon's purposes
qsirecon_spec = loads(load_data("io_spec.json").read_text())
bids_config = Config.load("bids")
deriv_config = Config.load("derivatives")

qsirecon_entities = {v["name"]: v["pattern"] for v in qsirecon_spec["entities"]}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **qsirecon_entities}
merged_entities = [{"name": k, "pattern": v} for k, v in merged_entities.items()]
config_entities = frozenset({e["name"] for e in merged_entities})


def get_bids_params(fullpath):
    bids_patterns = [
        r"^(.*/)?(?P<subject_id>sub-[a-zA-Z0-9]+)",
        "^.*_(?P<session_id>ses-[a-zA-Z0-9]+)",
        "^.*_(?P<task_id>task-[a-zA-Z0-9]+)",
        "^.*_(?P<acq_id>acq-[a-zA-Z0-9]+)",
        "^.*_(?P<space_id>space-[a-zA-Z0-9]+)",
        "^.*_(?P<rec_id>rec-[a-zA-Z0-9]+)",
        "^.*_(?P<run_id>run-[a-zA-Z0-9]+)",
        "^.*_(?P<dir_id>dir-[a-zA-Z0-9]+)",
    ]
    matches = {
        "subject_id": None,
        "session_id": None,
        "task_id": None,
        "dir_id": None,
        "acq_id": None,
        "space_id": None,
        "rec_id": None,
        "run_id": None,
    }
    for pattern in bids_patterns:
        pat = re.compile(pattern)
        match = pat.search(fullpath)
        params = match.groupdict() if match is not None else {}
        matches.update(params)
    return matches


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using xcp_d's configuration files.
    """

    out_path_base = ""
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = qsirecon_spec["default_path_patterns"]


def get_recon_output_name(
    base_dir,
    source_file,
    derivative_file,
    output_bids_entities,
    use_ext=True,
    qsirecon_suffix=None,
    dismiss_entities=None,
):
    source_entities = parse_file_entities(source_file)
    if dismiss_entities:
        source_entities = {k: v for k, v in source_entities.items() if k not in dismiss_entities}

    out_path = base_dir
    if qsirecon_suffix:
        out_path = op.join(out_path, f"qsirecon-{qsirecon_suffix}")

    # Infer the appropriate extension
    if "extension" not in output_bids_entities:
        ext = "." + ".".join(os.path.basename(derivative_file).split(".")[1:])
        output_bids_entities["extension"] = ext

    # Add the suffix
    output_bids_entities["suffix"] = output_bids_entities.get("suffix", "dwimap")

    # Add any missing entities from the source file
    output_bids_entities = {**source_entities, **output_bids_entities}

    out_filename = config.execution.layout.build_path(
        source=output_bids_entities,
        path_patterns=qsirecon_spec["default_path_patterns"],
        validate=False,
        absolute_paths=False,
    )
    if not use_ext:
        # Drop the extension from the filename
        out_filename = out_filename.split(".")[0]

    return os.path.join(out_path, out_filename)


class _ReconDerivativesDataSinkInputSpec(_DerivativesDataSinkInputSpec):
    in_file = traits.Either(
        traits.Directory(exists=True),
        InputMultiObject(File(exists=True)),
        mandatory=True,
        desc="the object to be saved",
    )
    mdp = traits.Str("", usedefault=True, desc="Label for model derived parameter field")
    mfp = traits.Str("", usedefault=True, desc="Label for model fit parameter field")
    model = traits.Str("", usedefault=True, desc="Label for model field")
    bundle = traits.Str("", usedefault=True, desc="Label for bundle field")
    bundles = traits.Str("", usedefault=True, desc="Label for bundles field")
    fit = traits.Str("", usedefault=True, desc="Label for fit field")
    label = traits.Str("", usedefault=True, desc="Label for label field")
    atlas = traits.Str("", usedefault=True, desc="Label for label field")
    qsirecon_suffix = traits.Str(
        "", usedefault=True, desc="name appended to qsirecon- in the derivatives"
    )


class ReconDerivativesDataSink(DerivativesDataSink):
    input_spec = _ReconDerivativesDataSinkInputSpec
    out_path_base = "qsirecon"

    def _run_interface(self, runtime):

        # If there is no qsirecon suffix, then we're not saving this file
        if not self.inputs.qsirecon_suffix:
            return runtime

        # Figure out what the extension should be based on the input file and compression
        src_fname, _ = _splitext(self.inputs.source_file)
        src_fname, dtype = src_fname.rsplit("_", 1)
        _, ext = _splitext(self.inputs.in_file[0])
        if self.inputs.compress is True and not ext.endswith(".gz"):
            ext += ".gz"
        elif self.inputs.compress is False and ext.endswith(".gz"):
            ext = ext[:-3]

        # Prepare the bids entities from the inputs
        output_bids = {}
        if self.inputs.atlas:
            output_bids["atlas"] = self.inputs.atlas
        if self.inputs.space:
            output_bids["space"] = self.inputs.space
        if self.inputs.bundles:
            output_bids["bundles"] = self.inputs.bundles
        if self.inputs.bundle:
            output_bids["bundle"] = self.inputs.bundle
        if self.inputs.space:
            output_bids["space"] = self.inputs.space
        if self.inputs.model:
            output_bids["model"] = self.inputs.model
        if self.inputs.mdp:
            output_bids["mdp"] = self.inputs.mdp
        if self.inputs.mfp:
            output_bids["mfp"] = self.inputs.mfp
        if self.inputs.fit:
            output_bids["fit"] = self.inputs.fit
        if self.inputs.suffix:
            output_bids["suffix"] = self.inputs.suffix
        if self.inputs.label:
            output_bids["label"] = self.inputs.label

        # Get the output name without an extension
        bname = get_recon_output_name(
            base_dir=self.inputs.base_directory,
            source_file=self.inputs.source_file,
            derivative_file=self.inputs.in_file[0],
            output_bids_entities=output_bids,
            use_ext=False,
        )

        # Ensure the directory exists
        os.makedirs(op.dirname(bname), exist_ok=True)

        formatstr = "{bname}{ext}"
        # If the derivative is a directory, copy it over
        copy_dir = op.isdir(str(self.inputs.in_file[0]))
        if copy_dir:
            out_file = formatstr.format(bname=bname, ext="")
            copytree(str(self.inputs.in_file), out_file, dirs_exist_ok=True)
            self._results["out_file"] = out_file
            return runtime

        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = "{bname}{i:04d}{ext}"

        # Otherwise it's file(s)
        self._results["compression"] = []
        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(bname=bname, i=i, ext=ext)
            if isdefined(self.inputs.extra_values):
                out_file = out_file.format(extra_value=self.inputs.extra_values[i])
            self._results["out_file"].append(out_file)
            self._results["compression"].append(_copy_any(fname, out_file))
        return runtime


def _splitext(fname):
    fname, ext = op.splitext(op.basename(fname))
    if ext == ".gz":
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext
    return fname, ext


def _copy_any(src, dst):
    src_isgz = src.endswith(".gz")
    dst_isgz = dst.endswith(".gz")
    if src_isgz == dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    dst_open = gzip.open if dst_isgz else open
    with src_open(src, "rb") as f_in:
        with dst_open(dst, "wb") as f_out:
            copyfileobj(f_in, f_out)
    return True
