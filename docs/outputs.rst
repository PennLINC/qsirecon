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

TODO: add Restriction Spectrum Imaging (RSI; :cite:p:`white2013rsi`) and Neurite Orientation Dispersion and Density Imaging (NODDI; :cite:p:`noddi`) below?

******************************
Diffusion tensor imaging (DTI)
******************************

The diffusion tensor model :cite:p:`basser1994a` provides a simple way to describe
Gaussian diffusion in a voxel.
The tensor model has been used extensively in the human brain :cite:p:`pierpaoli1996`
including developmental neuroscience, with large known effects of increasing FA and
decreasing MD with age :cite:p:`qiu2015`.
The eigenvectors and eigenvalues from the fitted tensor are used to calculate widely used
scalar maps such as fractional anisotropy (FA), mean diffusivity (MD), axial diffusivity (AD)
and radial diffusivity (RD).
When fitting tensors, we adopt the approach from the
`Adolescent Brain and Cognitive Development <https://abcdstudy.org/>`_
study :cite:p:`hagler2019` and perform multiple separate fits.
One fit used only the low-b (b≤1000) inner shells, where the assumptions of the tensor
model are most valid :cite:p:`desantis2011` and the results will be more similar to legacy
single shell studies. The second fit used all available data (e.g., "full shell") as in
:cite:p:`pines2020`. The inner shell tensor fit is computed twice: once in DSI Studio
using ordinary least squares and again in TORTOISE using weighted linear least
squares :cite:p:`basser1994b`, where tensor parameter estimation weights observations
by values proportional to their estimated SNR.
The full shell fit is only done in TORTOISE with weighted linear least squares.
Comparisons between full and inner shells should be done using the maps estimated by TORTOISE,
while comparisons of tensor fitting methods can be done with DSI Studio and TORTOISE inner
shell fits.

********************************
Diffusion kurtosis imaging (DKI)
********************************

Water diffusion in the brain is affected by the physical structures that make up neurons and organelles.
Instead of freely diffusing through space, water encounters barriers from myelin, cell membranes and other structures that introduce non-Gaussian features into the water diffusion distribution.
The Diffusion Kurtosis Imaging (DKI; :cite:p:`jensen2005dki`) model extends the of the DTI model by adding an additional 15 parameters that capture the deviations from Gaussianity missed when fitting the simple 6 parameter DTI model.
The DKI model incorporates data from all shells, potentially estimating the same scalar maps from DTI (FA, MD, etc) more accurately than a traditional tensor fit :cite:p:`henriques2021dki`.
In addition to the measures from DTI, the DKI model also allows one to compute additional scalars derived from the kurtosis tensor such as mean kurtosis (MK), radial kurtosis (RK), and axial kurtosis (AK) :cite:p:`jensen2010dki`.
DKI’s sensitivity to non-Gaussian diffusion makes it useful for capturing the interaction of water with more complex tissue features.


*************************************
Mean Apparent Propagator MRI (MAPMRI)
*************************************

The Mean Apparent Propagator (MAPMRI) method :cite:p:`ozarslan2013` is a model-free
approach that captures complex water diffusion. As a matter of practice, a diffusion tensor
is first computed (using just the inner shells (b<1250), saved as an output) to determine the
coordinate framework in which the ensemble average diffusion propagator (EAP) is to be estimated
in three dimensions by a combination of Hermite and Legendre polynomials. MAPMRI is estimated in
TORTOISE :cite:p:`irfanoglu2025` and maps are derived for multiple EAP-related properties.
One set of maps captures the probability of a water molecule returning to its origin (RTOP)
(which is inversely proportional to the pore size), to its principal axis (RTAP), or the plane
perpendicular to the principal axis (RTPP) (which is inversely proportional to the analog of
radial diffusivity). Furthermore, non-Gaussianity (NG) is calculated for the entire 3D, along
the principal direction of diffusion (NGPar) and perpendicular to it (NGPer).
The anisotropy of the EAP, or “propagator anisotropy” (PA). We calculate the angular difference,
𝜃, as the angular distance between the fitted MAPMRI coefficients and the coefficients
corresponding to its isotropic version :cite:p:`ozarslan2013`. Prior work in adolescents
and young adults has shown that MAPMRI scalars are robust to head motion and among the most
sensitive to age effects :cite:p:`pines2020`. Critically, our estimation of MAPMRI uses
the metadata present in BIDS to define the Large and Small delta diffusion-gradient timing
parameters (Δ and δ) for each scan.

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

.. bibliography::