.. include:: links.rst


##############################
Available Models in *QSIRecon*
##############################


Voxelwise models fits of dMRI data can be used to produce spatial maps where each
voxel’s value reflects a specific property of the diffusion process.

As each model often estimates several diffusion properties, at present these models
yield over 40 whole-brain parametric microstructure maps per dMRI imaging session.
Below we describe the five models that are fit as part of the qsirecon workflow and
their associated scalar maps:

.. toctree::
   :maxdepth: 1

   models/dti
   models/dki
   models/mapmri
   models/gqi
   models/noddi
