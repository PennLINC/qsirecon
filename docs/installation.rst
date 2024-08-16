.. include:: links.rst

.. _install_qsirecon:

############
Installation
############

The official source of QSIRecon is the `docker repository
<https://hub.docker.com/pennlinc/qsirecon>`_. The image

*******************************
Running QSIRecon via containers
*******************************


.. _install_docker`:

Docker
======

In order to run qsirecon in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_.

.. note:: If running Docker Desktop on MacOS (or via Docker Desktop), be sure to set
    the memory to 6 or more GB. Too little memory assigned to Docker Desktop can result
    in a message like ``Killed.``

A Docker command line may look like:

.. code-block:: bash

    docker run -ti --rm \
        -v /filepath/to/data/dir \
        -v /filepath/to/output/dir \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennlinc/qsirecon:latest \
        /filepath/to/data/dir /filepath/to/output/dir participant \
        --fs-license-file /opt/freesurfer/license.txt \
        --recon-spec pyafq_tractometry

.. _install_apptainer:

Singularity/Apptainer
=====================

The easiest way to get a Sigularity image is to run:

.. code-block:: bash

    apptainer build qsirecon-<version>.sif docker://pennlinc/qsirecon:<version>

Where ``<version>`` should be replaced with the desired version of qsirecon that you want to download.
Do not use ``latest`` or ``unstable`` unless you are performing limited testing.

As with Docker, you will need to bind the Freesurfer license.txt when running Singularity :

.. code-block:: bash

    apptainer run \
        --containall \
        --writable-tmpfs \
        -B "${PWD}","${FREESURFER_HOME}"/license.txt:/opt/freesurfer/license.txt \
        qsirecon-<version>.sif \
        "${PWD}"/derivatives/qsiprep \
        "${PWD}"/derivatives/qsirecon \
        participant \
        --fs-license-file /opt/freesurfer/license.txt \
        --recon-spec ss3t_autotrack \
        -w "${PWD}/work" \
        -v -v


External Dependencies
---------------------

QSIRecon is written using Python 3.10, and is based on nipype_.
The external dependencies are built in the
`qsirecon_build <https://github.com/PennLINC/qsirecon_build>`_ repository.
There you can find the URLs used to download the dependency source code
and the steps to compile each dependency.
