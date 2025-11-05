.. include:: links.rst

#####################
Outputs of *QSIRecon*
#####################

*QSIRecon* outputs are organized according to BIDS Derivatives standards.

.. admonition:: A note on BIDS compliance

   *QSIRecon* attempts to follow the BIDS specification as closely as possible.
   However, many *QSIRecon* derivatives are not currently covered by the specification.
   In those instances, we attempt to follow recommendations from existing BIDS Extension Proposals
   (BEPs), which are in-progress proposals to add new features to BIDS.
   However, we do not guarantee compliance with any BEP,
   as they are not yet part of the official BIDS specification.

   Three BEPs that are of particular use in *QSIRecon* are
   `BEP016: Diffusion weighted imaging derivatives <https://bids-website.readthedocs.io/en/latest/extensions/beps/bep_016.html>`_,
   `BEP017: Relationship & connectivity matrix data schema <https://docs.google.com/document/d/1ugBdUF6dhElXdj3u9vw0iWjE6f_Bibsro3ah7sRV0GA/edit?usp=sharing>`_,
   and
   `BEP038: Atlas Specification <https://docs.google.com/document/d/1RxW4cARr3-EiBEcXjLpSIVidvnUSHE7yJCUY91i5TfM/edit?usp=sharing>`_.

   In cases where a derivative type is not covered by an existing BEP,
   we have simply attempted to follow the general principles of BIDS.

   If you discover a problem with the BIDS compliance of *QSIRecon*'s derivatives,
   please open an issue in the *QSIRecon* repository.


******************************************
Whole-brain parametric microstructure maps
******************************************

Voxelwise models fits of dMRI data can be used to produce spatial maps where each voxel’s value reflects a specific property of the diffusion process.
The multi-shell dMRI acquisition lends itself to a wide range of such dMRI models, which have different strengths and challenges in measuring specific aspects of the diffusion process.
As each model often estimates several diffusion properties, at present these models yield over 40 whole-brain parametric microstructure maps per dMRI imaging session.
Below we describe the four models that are fit as part of the qsirecon workflow and their associated scalar maps.

TODO: add Restriction Spectrum Imaging (RSI; :cite:`white2013rsi`) and Neurite Orientation Dispersion and Density Imaging (NODDI; :cite:`noddi`) below?

******************************
Diffusion tensor imaging (DTI)
******************************



********************************
Diffusion kurtosis imaging (DKI)
********************************

Water diffusion in the brain is affected by the physical structures that make up neurons and organelles.
Instead of freely diffusing through space, water encounters barriers from myelin, cell membranes and other structures that introduce non-Gaussian features into the water diffusion distribution.
The Diffusion Kurtosis Imaging (DKI; :cite:`jensen2005dki`) model extends the of the DTI model by adding an additional 15 parameters that capture the deviations from Gaussianity missed when fitting the simple 6 parameter DTI model.
The DKI model incorporates data from all shells, potentially estimating the same scalar maps from DTI (FA, MD, etc) more accurately than a traditional tensor fit :cite:`henriques2021dki`.
In addition to the measures from DTI, the DKI model also allows one to compute additional scalars derived from the kurtosis tensor such as mean kurtosis (MK), radial kurtosis (RK), and axial kurtosis (AK) :cite:`jensen2010dki`.
DKI’s sensitivity to non-Gaussian diffusion makes it useful for capturing the interaction of water with more complex tissue features.


*************************************
Mean Apparent Propagator MRI (MAPMRI)
*************************************



************************************
Generalized q-Sampling Imaging (GQI)
************************************



********************
Dataset Organization
********************

*QSIRecon* organizes its outputs as self-contained BIDS-Derivative datasets.

In the output directory, you will find a derivative dataset containing anatomical and
subject-space atlas files, along with the `MRtrix3-specific renumbering <https://mrtrix.readthedocs.io/en/3.0.4/quantitative_structural_connectivity/labelconvert_tutorial.html>`_ files ::

   qsirecon/
      dataset_description.json
      logs/
      sub-<label>/[ses-<label>/]
         dwi/
            <source_entities>_seg-<atlas>_dseg.mif.gz
            <source_entities>_seg-<atlas>_dseg.nii.gz
            <source_entities>_seg-<atlas>_dseg.txt

Within the output directory, you will also find a "derivatives" directory containing
a separate self-contained derivatives dataset for each reconstruction method used by the
reconstruction workflow.

Each dataset will have its own set of HTML reports summarizing the results of the
associated reconstruction method.
Here is an example output structure from a reconstruction workflow with two reconstruction methods:
DKI and TORTOISE::

   qsirecon/
      derivatives/
         qsirecon-DKI/
            dataset_description.json
            sub-<label>[_ses-<label>].html
            logs/
            sub-<label>/[ses-<label>/]
               figures/
               dwi/
                  <source_entities>_model-dki_param-ad_dwimap.json
                  <source_entities>_model-dki_param-ad_dwimap.nii.gz
                  <source_entities>_model-dki_param-ak_dwimap.json
                  <source_entities>_model-dki_param-ak_dwimap.nii.gz
                  <source_entities>_model-dki_param-kfa_dwimap.json
                  <source_entities>_model-dki_param-kfa_dwimap.nii.gz
                  <source_entities>_model-dki_param-md_dwimap.json
                  <source_entities>_model-dki_param-md_dwimap.nii.gz
                  <source_entities>_model-dki_param-mk_dwimap.json
                  <source_entities>_model-dki_param-mk_dwimap.nii.gz
                  <source_entities>_model-dki_param-mkt_dwimap.json
                  <source_entities>_model-dki_param-mkt_dwimap.nii.gz
                  <source_entities>_model-dki_param-rd_dwimap.json
                  <source_entities>_model-dki_param-rd_dwimap.nii.gz
                  <source_entities>_model-dki_param-rk_dwimap.json
                  <source_entities>_model-dki_param-rk_dwimap.nii.gz
                  <source_entities>_model-tensor_param-fa_dwimap.json
                  <source_entities>_model-tensor_param-fa_dwimap.nii.gz
                  # Microstructural metrics calculated if wmti is True
                  <source_entities>_model-dkimicro_param-awf_dwimap.json
                  <source_entities>_model-dkimicro_param-awf_dwimap.nii.gz
                  <source_entities>_model-dkimicro_param-rde_dwimap.json
                  <source_entities>_model-dkimicro_param-rde_dwimap.nii.gz

         qsirecon-TORTOISE/
            dataset_description.json
            sub-<label>[_ses-<label>].html
            logs/
            sub-<label>/[ses-<label>/]
               figures/
               dwi/
                  <source_entities>_model-mapmri_param-ng_dwimap.json
                  <source_entities>_model-mapmri_param-ng_dwimap.nii.gz
                  <source_entities>_model-mapmri_param-ngpar_dwimap.json
                  <source_entities>_model-mapmri_param-ngpar_dwimap.nii.gz
                  <source_entities>_model-mapmri_param-ngperp_dwimap.json
                  <source_entities>_model-mapmri_param-ngperp_dwimap.nii.gz
                  <source_entities>_model-mapmri_param-pa_dwimap.json
                  <source_entities>_model-mapmri_param-pa_dwimap.nii.gz
                  <source_entities>_model-mapmri_param-path_dwimap.json
                  <source_entities>_model-mapmri_param-path_dwimap.nii.gz
                  <source_entities>_model-mapmri_param-rtap_dwimap.json
                  <source_entities>_model-mapmri_param-rtap_dwimap.nii.gz
                  <source_entities>_model-mapmri_param-rtop_dwimap.json
                  <source_entities>_model-mapmri_param-rtop_dwimap.nii.gz
                  <source_entities>_model-mapmri_param-rtpp_dwimap.json
                  <source_entities>_model-mapmri_param-rtpp_dwimap.nii.gz
                  <source_entities>_model-tensor_param-ad_dwimap.json
                  <source_entities>_model-tensor_param-ad_dwimap.nii.gz
                  <source_entities>_model-tensor_param-am_dwimap.json
                  <source_entities>_model-tensor_param-am_dwimap.nii.gz
                  <source_entities>_model-tensor_param-fa_dwimap.json
                  <source_entities>_model-tensor_param-fa_dwimap.nii.gz
                  <source_entities>_model-tensor_param-li_dwimap.json
                  <source_entities>_model-tensor_param-li_dwimap.nii.gz
                  <source_entities>_model-tensor_param-rd_dwimap.json
                  <source_entities>_model-tensor_param-rd_dwimap.nii.gz

If parcellation is enabled,
the output directory will also contain an "atlases" directory containing a BIDS-Atlas dataset.
See :ref:`connectivity_atlases` for more information on this output directory.


**********
References
**********

.. footbibliography::