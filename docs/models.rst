.. include:: links.rst

##############################
Available Models in *QSIRecon*
##############################

*QSIRecon* supports the following models:

* Diffusion Tensor Imaging (DTI)
* Diffusion Kurtosis Imaging (DKI)
* Mean Apparent Propagator MRI (MAPMRI)
* Generalized q-Sampling Imaging (GQI)


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

A key feature of the post-processing in the HBCD pipeline is the use of generalized
q-sampling imaging (GQI). GQI is another model-free approach that estimates water
diffusion orientation distribution functions (dODFs) using an analytic transform of
the diffusion signal :cite:p:`yeh2010`. Like the other models, GQI produces a number
of parametric microstructure maps such as generalized fractional anisotropy (GFA),
quantitative anisotropy (QA), and isotropic component (ISO) :cite:p:`yeh2013`.

We chose to use the peak directions of dODFs estimated via GQI as the basis for
tractography instead of the popular constrained spherical deconvolution (CSD) method
for a number of practical reasons. Although the HBCD dMRI acquisition meets the
criteria to use the multi-shell multi-tissue (MSMT) CSD method, the age range sampled
by HBCD makes it difficult to apply the method consistently and optimally.
For instance, MSMT-CSD works optimally on two tissue types in neonates
:cite:p:`pietsch2019,grotheer2022`, but typically three tissues are
included when analyzing images from adults :cite:p:`jeurissen2014`. Also, GQI does
not require a response function and is applied identically regardless of age.
Finally, while GQI has been used extensively with adult data over the last 15 years,
it has also been used successfully for tractography in infants :cite:p:`borchers2020,dennis2019,lee2021,barnesdavis2024` including data
from dHCP :cite:p:`sun2024`.

**********
References
**********

.. bibliography::
   :style: unsrt