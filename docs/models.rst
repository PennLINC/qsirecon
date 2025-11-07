.. include:: links.rst

##############################
Available Models in *QSIRecon*
##############################

******************************************
Whole-brain parametric microstructure maps
******************************************

Voxelwise models fits of dMRI data can be used to produce spatial maps where each 
voxel’s value reflects a specific property of the diffusion process.

*QSIRecon* supports the following models:

* Diffusion Tensor Imaging (DTI)
* Diffusion Kurtosis Imaging (DKI)
* Mean Apparent Propagator MRI (MAPMRI)
* Generalized q-Sampling Imaging (GQI)
* Neurite Orientation Dispersion and Density Imaging (NODDI)

As each model often estimates several diffusion properties, at present these models 
yield over 40 whole-brain parametric microstructure maps per dMRI imaging session.
Below we describe the five models that are fit as part of the qsirecon workflow and 
their associated scalar maps.

******************************
Diffusion tensor imaging (DTI)
******************************

DTI Foundational Papers
=======================

**DTI Concept Introduction**: Basser et al. introduced DTI as a new MRI 
modality that computes a diffusion tensor per voxel, yielding eigenvalues and 
eigenvectors that describe 3D water diffusion :cite:p:`basser1994`. This enabled mapping of 
fiber orientations in tissue and orientation-independent scalar measures like the 
trace (sum of eigenvalues, related to mean diffusivity). It established the idea 
that tensors encode more microstructural information than single-direction 
diffusivities.

**First Anisotropy Indices**: Pierpaoli and Basser defined fractional anisotropy (FA) 
– a normalized 0–1 index quantifying how anisotropic (directionally dependent) diffusion 
is :cite:p:`pierpaolibasser1996`. 
They demonstrated that earlier methods (diffusion measured in just a few directions) 
underestimated anisotropy, and introduced rotationally invariant metrics (FA, relative 
anisotropy, etc.) to robustly characterize white matter integrity. This work also 
cautioned that noise can bias anisotropy measures requiring eigenvalue ordering.

**First Human DTI Maps**: Pierpaoli et al. acquired the first DTI scans of the human 
brain, mapping principal diffusion directions and magnitudes :cite:p:`pierpaoli1996`. 
first DTI scans of the human brain, mapping principal diffusion directions and 
magnitudes. They observed that water diffuses ~3 times faster along axonal 
fibers than perpendicular to them in highly coherent tracts (e.g. corpus 
allosum), whereas regions with crossing or less organized fibers showed lower 
anisotropy. They also introduced Trace(D) (equal to 3×Mean Diffusivity (MD)) as 
an orientation-invariant measure of overall diffusivity, which was roughly 
uniform in normal brain except higher in cortical gray matter due to its 
higher water content

**Eigenvalue-Derived Metrics**: By the early 2000s, researchers began interpreting 
individual tensor eigenvalues. Song et al. first showed that 
axial diffusivity (AD) (diffusion along the primary eigenvector, λ1) and 
radial diffusivity (RD) (diffusion perpendicular to axons, mean of λ2 & λ3) 
can provide pathologically specific insights :cite:p:`song2002`. 
In a mouse model of demyelination, RD increased with myelin loss while AD stayed 
constant (since axons remained intact). This seminal finding 
established that increases in RD selectively indicate myelin degeneration, whereas 
indicate myelin degeneration, whereas decreases in AD are more tied to axonal 
injury – a distinction that has since informed numerous neuroimaging studies 
of white matter diseases :cite:p:`song2002`.

DTI Changes Across The Lifespan
===============================

The tensor model has been used extensively in the human brain 
:cite:p:`pierpaoli1996` including developmental neuroscience, with large known 
effects of increasing FA and decreasing MD with age :cite:p:`qiu2015`.

**Neonatal Diffusivity**:  Neil et al. reported neonatal 
mean diffusivity values 1.5–2 times higher than in adults, with very low 
white-matter anisotropy :cite:p:`neil1998`. This reflects abundant free water and 
unmyelinated fibers at birth. Diffusivity drops and anisotropy rises steeply 
in the first postnatal months as the brain matures (water compartmentalizes and 
myelination progresses).

**Childhood to Adolescence**: White matter development continues through 
childhood and the teen years. Longitudinal data showed steady FA increases 
and MD (mean diffusivity) decreases in virtually all major tracts from age 5 
into the 20s :cite:p:`lebel2011`. Not all tracts mature simultaneously: 
early-developing motor/sensory pathways (e.g., internal capsule) reach adult-like 
FA by late adolescence, whereas association tracts in frontal and temporal lobes 
keep increasing in FA (and decreasing in RD) into the third decade. This prolonged 
maturation of frontal circuitry aligns with functional development of executive 
and cognitive abilities in late adolescence.

**Whole Lifespan Trajectories**: Cross-sectional analyses across the lifespan find 
that FA follows an "inverted U" trajectory: increasing from childhood to a peak 
in the 20s–30s, then declining with older age. In a sample of 430 subjects 
aged 8–85 :cite:p:`westlye2010`, fractional anisotropy plateaus by the early 30s 
and slowly falls thereafter, while mean and radial diffusivities do the reverse 
(minimal in young adults, then rising in aging). Interestingly, this large study 
found no simple "last-in-first-out" pattern although late-maturing frontal tracts 
often showed pronounced aging changes, all regions eventually exhibited 
microstructural decline, indicating a widespread but heterogeneous aging effect 
rather than one specific sequence :cite:p:`westlye2010`.

**Regional Patterns in Aging**: Many DTI studies of aging report that anterior white 
matter tracts (which myelinate last) are more vulnerable to aging. Salat et al. 
found significantly lower FA in older adults (mean age ~67) compared to young 
(mean ~24), especially in the frontal lobes and corpus callosum :cite:p:`salat2005`. 
In contrast, posterior tracts like the splenium of the callosum or occipital white 
matter showed smaller FA differences. Such findings support that age-related myelin 
degeneration and fiber loss are often greatest in late-developing, more complex pathways 
(though subsequent research has refined this view with more nuanced patterns).

**Microstructural Changes with Aging**: DTI metrics suggest that aging involves 
loss of fiber integrity (lower FA) and increased water mobility 
(higher MD/apparent diffusion coefficient (ADC)). 
In older adults, increased radial diffusivity is commonly observed, consistent 
with demyelination or degraded myelin packing, while axial diffusivity may also 
eventually decrease if axonal loss occurs. Longitudinal studies in elderly cohorts 
(e.g. over 60) have confirmed ongoing within-person FA declines annually 
:cite:p:`westlye2010`. These DTI changes correlate with cognitive slowing and 
executive function decline in many studies, highlighting DTI's value in tracking 
brain aging and its cognitive consequences.

DTI Methodological Warnings and Caveats
=======================================

**Single Tensor Limitations (Crossing Fibers)**: The basic DTI model assumes one 
dominant fiber orientation per voxel – an assumption often violated in the brain. 
In regions with crossing, kissing, or branching fibers, the tensor model yields 
an average that can underestimate anisotropy and obscure fiber directions. For 
instance, a voxel containing two crossing tracts will show an artificially low 
FA (appearing "isotropic") even if each tract is highly anisotropic. This issue 
can lead to misinterpretation of reduced FA: it might reflect complex fiber 
geometry rather than neural degeneration. Advanced high-angular-resolution methods 
(e.g., multi-tensor or Q-ball imaging) are recommended when crossing fibers are 
prevalent, or one should interpret DTI metrics in such regions with caution 
:cite:p:`wheelerkingshott2009`.

**Axial vs. Radial Diffusivity Interpretations**: While increases in radial 
diffusivity and decreases in axial diffusivity have been linked to demyelination 
and axonal injury respectively, one must be careful not to over-interpret these 
metrics in isolation. Wheeler-Kingshott and Cercignani showed that in voxels with 
multiple fiber orientations, a change in one eigenvalue 
can induce a "fictitious" change in another :cite:p:`wheelerkingshott2009`. 
For example, crossing fibers can make AD appear reduced even without axonal damage. 
Similarly, heavy pathology can alter the principal eigenvector direction, 
invalidating the simple mapping of λ1 to one fiber population. 
*Bottom line*: AD and RD are informative only in contexts where a single fiber 
population dominates the voxel; otherwise, observed changes might result 
from geometry or partial volume effects rather than specific 
histopathology :cite:p:`wheelerkingshott2009`.

**Partial Volume and Free Water Contamination**: DTI metrics can be skewed by 
mixing of tissue with free water (cerebrospinal fluid or edema). A small amount 
of free water in a voxel drastically lowers FA and raises diffusivity, since free 
water diffusion is fast and isotropic. This can mask true tissue changes – for 
example, a remyelinating lesion adjacent to CSF might still show low FA due to 
CSF contamination. Methods like the free-water elimination model 
:cite:p:`pasternak2009`address this by fitting a two-component model, 
effectively stripping out the isotropic diffusion component and revealing the 
"true" tissue tensor. Researchers should be mindful of partial voluming, especially 
in periventricular areas, and consider correction strategies or region-of-interest 
approaches to avoid artifactual findings.

**Noise, Motion, and Artifacts**: DTI outcomes are sensitive to data quality. 
Thermal noise in diffusion-weighted images leads to bias (e.g., a noise floor 
artificially boosts low ADC values), and insufficient signal-to-noise can make 
FA appear higher in low-FA regions (background noise imposes a floor) 
:cite:p:`jones2010`. Head motion is another major concern: even subtle 
motion can cause directional-dependent blurring or signal drop-out, which may 
mimic or mask true diffusion anisotropy. For example, uncorrected motion can 
spuriously increase FA in gray matter or produce group differences unrelated to 
biology (as noted in studies of populations like children or patients who move 
more) :cite:p:`yendiki2014`. Best practices include using motion correction 
algorithms, excluding data with excessive motion, and using robust fitting methods 
(e.g., iteratively reweighted least squares) that down-weight outliers caused by 
artifacts.

**Acquisition and Analysis Choices**: The choice of diffusion gradient directions 
and b-value can influence DTI metrics. A minimal 6-direction tensor encoding is 
insufficient for reliable quantitative work – more directions (20–30+) are 
recommended to stabilize FA/MD measures and reduce variability. Similarly, 
moderate b-values (~1000 s/mm²) are typically chosen to balance SNR and 
sensitivity; very high b-values can introduce bias in tensor-fitting (and 
require models beyond DTI). During analysis, image alignment (registration) 
and smoothing can also introduce caveats: misregistration across subjects can 
blur tract-specific values, and heavy smoothing can artificially increase FA 
in partial volume voxels. Tools like tract-based spatial statistics (TBSS) were 
developed to mitigate some of these issues by skeletonizing white matter maps 
to focus on centers of tracts. The key caveat is that DTI analyses involve many 
processing steps, each of which must be done carefully – otherwise, errors can 
propagate and lead to incorrect conclusions. Community guidelines and 
detailed "pitfall" checklists (e.g., :cite:p:`jones2010`) are valuable 
resources to ensure methodological rigor in DTI studies.

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
   :filter: cited