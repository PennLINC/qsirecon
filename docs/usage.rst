.. include:: links.rst

Usage
-----

We strongly recommend running ``qsirecon`` from a container.
The common parts of the command are similar to the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition, but the input should
*not* be a BIDS dataset, but rather one of the supported preprocessed
inputs (qsiprep, or UKBB results).

Suppose I'm in a directory where there are some qsiprep results in
``inputs/qsiprep``. I'd like to save my ``qsirecon`` outputs in ``results``. I
have access to 8 cpus.  To run the ``dsi_studio_autotrack`` workflow from
``qsirecon-latest.sif`` I could use:

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
