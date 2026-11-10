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
The ``--output-spaces`` option names the template space that scalar maps are warped into
(in workflows that map scalars to a template).
It does not affect atlases, which are only written in the space of the DWI
(see :ref:`atlas_templates`).
It follows the same syntax as *QSIPrep*'s ``--output-spaces``:
a TemplateFlow template name, optionally followed by a cohort and a resolution::

    --output-spaces MNI152NLin2009cAsym
    --output-spaces MNI152NLin2009cAsym:res-2mm
    --output-spaces MNIInfant:cohort-2:res-1p5mm

When the option is not given, ``MNI152NLin2009cAsym`` is used.

*QSIRecon* does not estimate a normalization.
It uses the transforms between the anatomical space and the requested template that
are in the input dataset (``from-ACPC_to-<template>_mode-image_xfm.h5`` and its inverse),
which *QSIPrep* writes for each template it was asked for.
Only one output space may be requested, and the DWI spaces that *QSIPrep* accepts
(``acpc``, ``anat`` and ``dwiref``) are not valid here.
To choose which preprocessed DWI files are reconstructed, use ``--bids-filter-file``.

Spaces the data were not normalized to
======================================

If the input dataset has no transforms for the requested template,
*QSIRecon* looks for a chain of transforms that reaches it:
the transforms in the input dataset, followed by the template-to-template transforms
that `TemplateFlow <https://www.templateflow.org>`_ distributes.
For example, if *QSIPrep* was run with ``--output-spaces acpc MNI152NLin2009cAsym``,
then ``--output-spaces MNI152NLin6Asym`` uses

- ``from-ACPC_to-MNI152NLin2009cAsym`` (and its inverse) from the input dataset, and
- ``tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym`` (and its inverse) from TemplateFlow.

The transforms are passed to ANTs together, so the data are still interpolated once.
A few things to keep in mind:

- The shortest chain is used, and it may pass through more than one template.
  A warning in the log, and the methods boilerplate, name the templates it goes through.
- Each transform on the chain has its own registration error, so a chain is less
  accurate than normalizing to the template directly in *QSIPrep*.
- The template-to-template transforms are downloaded from TemplateFlow when the
  workflow is built, so this needs network access unless they are already in
  ``$TEMPLATEFLOW_HOME``.
- A transform is needed in both directions for every step of the chain.
- ``cohort-auto`` cannot be used for a space that is reached through a chain;
  name the cohort.

If no chain reaches the requested template, *QSIRecon* stops with an error that lists
the templates the input dataset has transforms for.

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

Cohorts
=======

Templates that are divided into cohorts, such as ``MNIInfant``, need one.
Name the cohort that *QSIPrep* used (``MNIInfant:cohort-2``),
or use ``cohort-auto`` to take whichever cohort the transforms in the input dataset were
estimated with.
*QSIRecon* does not choose a cohort from the participant's age.
If the input dataset has transforms for more than one cohort, ``cohort-auto`` is
ambiguous and the cohort must be named.

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
