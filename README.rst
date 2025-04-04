.. include:: links.rst

##############################################################
QSIRecon: Reconstruction of preprocessed q-space images (dMRI)
##############################################################

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

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14193616.svg
  :target: https://doi.org/10.5281/zenodo.14193616
  :alt: Zenodo DOI

.. image:: https://img.shields.io/badge/License-BSD--3--Clause-green
  :target: https://opensource.org/licenses/BSD-3-Clause
  :alt: License


Full documentation at https://qsirecon.readthedocs.io

*****
About
*****

QSIRecon builds post-processing workflows that produce many of the biologically-interesting dMRI
derivatives used for hypothesis testing. The main goal of QSIRecon is to make the state-of-the-art
methods available in Dipy_, MRTrix_, `DSI Studio`_, PyAFQ_  and other software packages easy to apply on
preprocessed dMRI data. QSIRecon is companion software for `XCP-D <https://xcp-d.readthedocs.io>`_,
doing for dMRI what XCP-D does for BOLD.

QSIRecon workflows can produce outputs such as

 * ODF/FOD reconstruction
 * Model fits and parameter estimation
 * Tractography
 * Tractometry
 * Regional connectivity
 * Tabular data

.. image:: https://github.com/PennLINC/qsirecon/raw/main/docs/_static/workflow_full.png

***************
Citing QSIRecon
***************

If you use QSIRecon in your research, please use the boilerplate generated by the workflow.
The main citation is

  Cieslak, M., Cook, P. A., He, X., Yeh, F. C., Dhollander, T., Adebimpe, A.,
  ... & Satterthwaite, T. D. (2021). QSIPrep: an integrative platform for preprocessing
  and reconstructing diffusion MRI data. Nature methods, 18(7), 775-778.
