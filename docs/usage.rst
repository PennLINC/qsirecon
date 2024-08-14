.. include:: links.rst

Usage
-----

Before you can use QSIRecon, you must have some preprocessed dMRI.
*Raw BIDS dMRI should not be used and will not work*. Instead,
gather data processed by QSIPrep_ or UKBioBank_. The directory
containing the ``sub-*`` directories (if from QSIPrep) or the
directory with the numerals/underscores per subject if UKBB
will be the first argument to QSIRecon.

The next step is to get a containerized version of QSIRecon. This can be
done with Singularity_, Apptainer_ or Docker_. Most users run QSIRecon on
a high performance computing cluster, so we will assume Apptainer is being
used throughout this documentation. See :ref:`Installation` on how to create
a `sif` file or pull the image with Docker.

Next, you need to decide which workflow you'd like to run. You can pick
from any of the :ref:`builtin_reconstruction` or :ref:`custom_reconstruction`.
Here we'll pick the ``dsi_studio_autotrack`` workflow.

Finally, you'll need to craft a command to set up your QSIRecon run.
Suppose you're in a directory where there are some qsiprep results in
``inputs/qsiprep``. You'd like to save QSIRecon outputs in ``results``. You
have access to 8 cpus.  To run the from ``qsirecon-latest.sif`` you could use:

Apptainer Example: ::

   apptainer run \
       --containall \
       --writable-tmpfs \
       -B "${PWD}" \
       "${PWD}/inputs/qsiprep" \
       "${PWD}/results" \
       participant \
       -w "${PWD}/work" \
       --nthreads 8 \
       --omp-nthreads 8 \
       --recon-spec dsi_studio_autotrack


Command-Line Arguments
======================

.. argparse::
   :ref: qsirecon.cli.parser._build_parser
   :prog: qsirecon
   :nodefault:
   :nodefaultconst:


Debugging
=========

Logs and crashfiles are outputted into the
``<output dir>/qsirecon/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.
