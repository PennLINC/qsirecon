.. include:: links.rst

QSIRecon: Preprocessing and analysis of q-space images
=======================================================

.. image:: https://img.shields.io/badge/Source%20Code-pennlinc%2Fqsirecon-purple
  :target: https://github.com/PennLINC/qsirecon
  :alt: GitHub Repository

.. image:: https://readthedocs.org/projects/qsirecon/badge/?version=latest
  :target: http://qsirecon.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/docker-pennlinc/qsirecon-brightgreen.svg?logo=docker&style=flat
  :target: https://hub.docker.com/r/pennlinc/qsirecon/tags/
  :alt: Docker

.. image:: https://circleci.com/gh/PennLINC/qsirecon/tree/master.svg?style=svg
  :target: https://circleci.com/gh/PennLINC/qsirecon/tree/master
  :alt: Test Status

.. image:: https://img.shields.io/badge/Nature%20Methods-10.1038%2Fs41592--021--01185--5-purple
  :target: https://doi.org/10.1038/s41592-021-01185-5
  :alt: Publication DOI

.. image:: https://img.shields.io/badge/License-BSD--3--Clause-green
  :target: https://opensource.org/licenses/BSD-3-Clause
  :alt: License


Full documentation at https://qsirecon.readthedocs.io

About
-----

``qsirecon`` configures pipelines for processing diffusion-weighted MRI (dMRI) data.
The main features of this software are

  1. A BIDS-app approach to preprocessing nearly all kinds of modern diffusion MRI data.
  2. Automatically generated preprocessing pipelines that correctly group, distortion correct,
     motion correct, denoise, coregister and resample your scans, producing visual reports and
     QC metrics.
  3. A system for running state-of-the-art reconstruction pipelines that include algorithms
     from Dipy_, MRTrix_, `DSI Studio`_  and others.
  4. A novel motion correction algorithm that works on DSI and random q-space sampling schemes

.. image:: https://github.com/PennLINC/qsirecon/raw/master/docs/_static/workflow_full.png


.. _preprocessing_def:

Preprocessing
~~~~~~~~~~~~~~~

The preprocessing pipelines are built based on the available BIDS inputs, ensuring that fieldmaps
are handled correctly. The preprocessing workflow performs head motion correction, susceptibility
distortion correction, MP-PCA denoising, coregistration to T1w images, spatial normalization
using ANTs_ and tissue segmentation.


.. _reconstruction_def:

Reconstruction
~~~~~~~~~~~~~~~~

The outputs from the :ref:`preprocessing_def` pipelines can be reconstructed in many other
software packages. We provide a curated set of :ref:`recon_workflows` in ``qsirecon``
that can run ODF/FOD reconstruction, tractography, Fixel estimation and regional
connectivity.


Note
------

The ``qsirecon`` pipeline uses much of the code from ``FMRIPREP``. It is critical
to note that the similarities in the code **do not imply that the authors of
FMRIPREP in any way endorse or support this code or its pipelines**.
