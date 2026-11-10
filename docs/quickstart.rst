.. include:: links.rst

###########
Quick Start
###########

Before you can use *QSIRecon*, you must have some preprocessed dMRI data.
See :ref:`input_data`

The next step is to get a containerized version of *QSIRecon*. This can be
done with Singularity_, Apptainer_ or Docker_. Most users run *QSIRecon* on
a high performance computing cluster, so we will assume Apptainer is being
used throughout this documentation. See :ref:`install_qsirecon` on how to create
a `sif` file or pull the image with Docker.

Next, you need to decide which workflow you'd like to run. You can pick
from any of the :ref:`builtin_workflows` or :ref:`building_workflows`.
Here we'll pick the ``dsi_studio_autotrack`` workflow.

Finally, you'll need to craft a command to set up your *QSIRecon* run.
Suppose you're in a directory where there are some qsiprep results in
``inputs/qsiprep``. You'd like to save *QSIRecon* outputs in ``results``. You
have access to 8 cpus.  To run the from ``qsirecon-latest.sif`` you could use:

.. code-block:: bash

   apptainer run \
       --containall \
       --writable-tmpfs \
       -B "${PWD}" \
       qsirecon-latest.sif \
       "${PWD}/inputs/qsiprep" \
       "${PWD}/results/qsirecon" \
       participant \
       -w "${PWD}/work" \
       --nthreads 8 \
       --omp-nthreads 8 \
       --recon-spec dsi_studio_autotrack \
       -v -v

Once this completes you will see a number of new directories written to ``results``.
You will find errors (if any occurred) and configuration files for each subject
directly under ``results/sub-*``. Each analysis also creates its own directory that
contains results per subject. In the case of ``dsi_studio_autotrack`` we will see
``results/qsirecon-DSIStudio/sub-*`` containing the outputs from the ss3t_autotrack
workflow. Some workflows produce multiple directories, particularly when multiple
models are fit.

.. _output_spaces:

*************
Output spaces
*************

*QSIRecon* always writes reconstruction outputs in the space of the preprocessed DWI,
on the grid of the input data.
The ``--output-spaces`` option names the template spaces that scalar maps are warped into
(in workflows that map scalars to a template).
It does not affect atlases, which are only written in the space of the DWI
(see :ref:`atlas_templates`).
It follows the same syntax as *QSIPrep*'s ``--output-spaces``:
a TemplateFlow template name, optionally followed by a cohort and one or more
resolutions::

    --output-spaces MNI152NLin2009cAsym
    --output-spaces MNI152NLin2009cAsym:res-2mm
    --output-spaces MNIInfant:cohort-2:res-1p5mm

When the option is not given, ``MNI152NLin2009cAsym`` is used.

Several spaces may be requested, and a space may be requested at several resolutions.
Scalar maps are written once for each::

    --output-spaces MNI152NLin2009cAsym:res-1:res-2mm MNI152NLin6Asym

writes each scalar map three times: in ``MNI152NLin2009cAsym`` on the template's
``res-1`` grid and with 2 mm voxels, and in ``MNI152NLin6Asym`` at the voxel size of the
preprocessed DWI.
A space and resolution that is requested twice is written once.

*QSIRecon* does not estimate a normalization.
It uses the transform from the anatomical space to each requested template that
is in the input dataset (``from-ACPC_to-<template>_mode-image_xfm.h5``),
which *QSIPrep* writes for each template it was asked for.
The DWI spaces that *QSIPrep* accepts (``acpc``, ``anat`` and ``dwiref``) are not valid
here.
To choose which preprocessed DWI files are reconstructed, use ``--bids-filter-file``.

Spaces the data were not normalized to
======================================

If the input dataset has no transform to a requested template,
*QSIRecon* looks for a chain of transforms that reaches it:
a transform in the input dataset, followed by the template-to-template transforms
that `TemplateFlow <https://www.templateflow.org>`_ distributes.
For example, if *QSIPrep* was run with ``--output-spaces acpc MNI152NLin2009cAsym``,
then ``--output-spaces MNI152NLin6Asym`` uses

- ``from-ACPC_to-MNI152NLin2009cAsym`` from the input dataset, and
- ``tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym`` from TemplateFlow.

The transforms are passed to ANTs together, so the data are still interpolated once.
A few things to keep in mind:

- Each requested space is resolved on its own, so one request can mix spaces the data
  were normalized to with spaces that are reached through a chain.
- The shortest chain is used, and it may pass through more than one template.
  A warning in the log, and the methods boilerplate, name the templates it goes through.
- Each transform on the chain has its own registration error, so a chain is less
  accurate than normalizing to the template directly in *QSIPrep*.
- The template-to-template transforms are downloaded from TemplateFlow when the
  workflow is built, so this needs network access unless they are already in
  ``$TEMPLATEFLOW_HOME``.
- Scalar maps only go from the anatomical space into a template, so a chain only needs
  the transforms that point that way.
  The transforms out of a template are not used for output spaces.
- ``cohort-auto`` cannot be used for a space that is reached through a chain;
  name the cohort.

If no chain reaches a requested template, *QSIRecon* stops with an error that lists
the templates the input dataset has transforms to.
This is the case even when the other requested spaces can be reached:
no space is skipped.

The regions shown in the ODF figures of the reports are drawn in ``MNI152NLin2009cAsym``,
so they are always brought into the space of the DWI from that template, in the same way:
with the input dataset's own transform when it has one, and through a chain otherwise.
If no chain is found or a transform cannot be downloaded, a warning is logged and the
ODF figures are left out of the reports.

Resolutions
===========

The resolution sets the voxel size of the template-space outputs:

- ``res-<size>mm`` is a physical voxel size, with ``p`` as the decimal point:
  ``res-2mm``, ``res-1p5mm``, or the anisotropic ``res-1p5x1p5x3mm``.
- ``res-<label>`` is a TemplateFlow resolution label, such as ``res-2``.
  The template is fetched at that resolution, and its voxel size is kept.
- With no resolution, the smallest voxel dimension of the preprocessed DWI is used.

If the requested voxel size is smaller than that of the preprocessed DWI, the data are
upsampled.

A resolution is written to the names of the template-space scalar maps as the ``res``
entity, exactly as it was requested, so that several resolutions of one template can
sit side by side::

    sub-01_space-MNI152NLin2009cAsym_res-1_model-dki_param-fa_dwimap.nii.gz
    sub-01_space-MNI152NLin2009cAsym_res-2mm_model-dki_param-fa_dwimap.nii.gz
    sub-01_space-MNI152NLin6Asym_model-dki_param-fa_dwimap.nii.gz

A space that is requested without a resolution has no ``res`` entity in its file names.
A ``res`` entity in the name of the preprocessed DWI is never carried over to the
outputs, since it describes the grid of the input data.

Cohorts
=======

Templates that are divided into cohorts, such as ``MNIInfant``, need one.
Name the cohort that *QSIPrep* used (``MNIInfant:cohort-2``),
or use ``cohort-auto`` to take whichever cohort the transforms in the input dataset were
estimated with.
*QSIRecon* does not choose a cohort from the participant's age.
If the input dataset has transforms for more than one cohort, ``cohort-auto`` is
ambiguous and the cohort must be named.
``cohort-auto`` is replaced with the cohort it stands for before the requested spaces
are compared, so ``MNIInfant:cohort-2 MNIInfant:cohort-auto`` is one space for data that
were normalized to cohort 2.

Requesting an infant template (``MNIInfant`` or ``UNCInfant``) also tightens the field of
view of the template-space outputs.

Removed options
===============

``--output-spaces`` replaces two options:

.. list-table::
   :header-rows: 1

   * - Removed option
     - Equivalent request
   * - ``--output-resolution 2``
     - ``--output-spaces MNI152NLin2009cAsym:res-2mm``
   * - ``--infant``
     - ``--output-spaces MNIInfant:cohort-auto``
   * - ``--infant --output-resolution 1.5``
     - ``--output-spaces MNIInfant:cohort-auto:res-1p5mm``


**********************
Command-Line Arguments
**********************

.. argparse::
   :ref: qsirecon.cli.parser._build_parser
   :prog: qsirecon
   :func: _build_parser

***************
Troubleshooting
***************

Logs and crashfiles are outputted into the
``<output dir>/qsirecon/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.
