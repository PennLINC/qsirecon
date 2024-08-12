.. include:: links.rst

QSIRecon: Reconstruction of preprocessed q-space images
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

.. image:: https://circleci.com/gh/PennLINC/qsirecon/tree/main.svg?style=svg
  :target: https://circleci.com/gh/PennLINC/qsirecon/tree/main
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

  1. A system for running state-of-the-art reconstruction pipelines that include algorithms
     from Dipy_, MRTrix_, `DSI Studio`_  and others.

.. image:: https://github.com/PennLINC/qsirecon/raw/main/docs/_static/workflow_full.png


.. _reconstruction_def:

Reconstruction
~~~~~~~~~~~~~~~~

The outputs from BIDS-compliant preprocessing pipelines can be reconstructed in many other
software packages. We provide a curated set of :ref:`recon_workflows` in ``qsirecon``
that can run ODF/FOD reconstruction, tractography, Fixel estimation and regional
connectivity.


Note
----

The ``QSIRecon`` pipeline uses much of the code from ``fMRIPrep``.
It is critical to note that the similarities in the code
**do not imply that the authors of fMRIPrep in any way endorse or support this code or its pipelines**.
