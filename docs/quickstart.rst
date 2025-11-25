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

Next, you need to decide which pipeline you'd like to run. You can pick
from any of the :ref:`builtin_pipelines` or :ref:`building_pipelines`.
Here we'll pick the ``dsi_studio_autotrack`` pipeline.

Finally, you'll need to craft a command to set up your *QSIRecon* run.
Suppose you're in a directory where there are some qsiprep results in
``inputs/qsiprep``. You'd like to save *QSIRecon* outputs in ``results``. You
have access to 8 cpus.  To run the from ``qsirecon-latest.sif`` you could use:

.. code-block:: bash

   apptainer run \
       --containall \
       --writable-tmpfs \
       -B "${PWD}" \
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
pipeline. Some pipelines produce multiple directories, particularly when multiple
models are fit.

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
