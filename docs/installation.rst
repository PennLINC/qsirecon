.. include:: links.rst

------------
Installation
------------

There are two easy ways to use qsirecon:
in a `Docker Container`_, or in a `Singularity Container`_.
Using a local container method is highly recommended.


.. _`Docker Container`:

Docker Container
================

In order to run qsirecon in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_.

.. note:: If running Docker Desktop on MacOS (or via Docker Desktop), be sure to set
    the memory to 6 or more GB. Too little memory assigned to Docker Desktop can result
    in a message like ``Killed.``

A Docker command line may look like::

    $ docker run -ti --rm \
        -v /filepath/to/data/dir \
        -v /filepath/to/output/dir \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennlinc/qsirecon:latest \
        /filepath/to/data/dir /filepath/to/output/dir participant \
        --fs-license-file /opt/freesurfer/license.txt \
        --recon-spec pyafq_tractometry


Singularity/Apptainer Container
===============================

The easiest way to get a Sigularity image is to run::

    $ singularity build qsirecon-<version>.sif docker://pennlinc/qsirecon:<version>

Where ``<version>`` should be replaced with the desired version of qsirecon that you want to download.
Do not use ``latest`` or ``unstable`` unless you are performing limited testing.

As with Docker, you will need to bind the Freesurfer license.txt when running Singularity ::

    $ singularity run --containall --writable-tmpfs \
        -B $HOME/qsiprep_outputs,$HOME/dockerout,${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        qsirecon-<version>.sif \
        $HOME/qsiprep_outputs $HOME/dockerout participant \
        --fs-license-file /opt/freesurfer/license.txt \
        --recon-spec ss3t_autotrack


External Dependencies
---------------------

QSIRecon is written using Python 3.10, and is based on nipype_.
The external dependencies are built in the
`qsirecon_build <https://github.com/PennLINC/qsirecon_build>`_ repository.
There you can find the URLs used to download the dependency source code
and the steps to compile each dependency.
