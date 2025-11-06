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

Foundational Papers and First Derivations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- DKI was first introduced by Jensen et al. (2005) as an extension of DTI to measure **diffusion non-Gaussianity** in tissues :cite:p:`jensen2005dki`.
Their seminal work defined **diffusional kurtosis** as a quantitative marker, showing that normal white matter has substantially higher kurtosis than gray matter (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/15906300/#:~:text=provides%20an%20estimate%20for%20the,with%20a%20variety%20of%20neuropathologies>`_).
- Building on this, Lu et al. (2006) provided the first full **mathematical derivation of the kurtosis tensor**, introduced rotational invariants like **mean kurtosis (MK)** (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/16521095/#:~:text=Conventional%20diffusion%20tensor%20imaging%20,space>`_), reported reproducible MK values, and showed that kurtosis anisotropy can reveal complex fiber geometries (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/16521095/#:~:text=,information%20to%20that%20of%20DTI>`_).
- Jensen and Helpern’s 2010 review consolidated the DKI model, formalizing **MK, AK, RK** as rotationally invariant descriptors (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/20632416/#:~:text=Quantification%20of%20non,This%20review%20discusses>`_), discussed practical acquisition requirements (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/20632416/#:~:text=compartments.%20The%20degree%20of%20non,that%20the%20diffusional%20kurtosis%20is>`_) and highlighted DKI’s sensitivity to tissue heterogeneity (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/20632416/#:~:text=the%20underlying%20theory%20of%20DKI,as%20Alzheimer%27s%20disease%20and%20schizophrenia>`_).
- Tabesh et al. (2011) introduced **constrained least-squares estimation** to ensure physically valid DKI fits (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/21337412/#:~:text=imaging%20artifacts,The>`_), defined **KFA**, and provided closed-form formulas for MK and RK (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/21337412/#:~:text=The%20advantage%20offered%20by%20the,form%20formulae>`_).
- Fieremans et al. (2011) extended DKI toward **microstructural modeling of white matter**, deriving **AWF** and **extra-axonal tortuosity** from DKI data (`pmc.ncbi.nlm.nih.gov <https://pmc.ncbi.nlm.nih.gov/articles/PMC3136876/#:~:text=tissues%20using%20magnetic%20resonance%20imaging,be%20determined%20directly%20from%20the>`_) and aligning DKI-derived parameters with known tissue features (`pmc.ncbi.nlm.nih.gov <https://pmc.ncbi.nlm.nih.gov/articles/PMC3136876/#:~:text=diffusion%20metrics%20conventionally%20obtained%20with,important%20information%20on%20neurodegenerative%20disorders>`_).

TODO Add: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.675433/full

Influential Lifespan Findings (Development and Aging)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Normal aging:** Falangola et al. (2008) showed age-related changes in DKI metrics across the healthy lifespan; MD increased and FA decreased in the oldest group (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/19025941/#:~:text=Results%3A%20%20We%20found%20significant,MD%20and%20decrease%20of%20FA>`_), while **MK** exhibited distinct trends across the lifespan (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/19025941/#:~:text=with%20increase%20of%20MD%20and,decrease%20of%20FA>`_).
- **Early development:** Paydar et al. (2014) demonstrated that **FA and MK rise with age in WM**, but MK continues to increase after FA plateaus; MK also revealed GM maturation undetectable by FA (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/24231848/#:~:text=Results%3A%20%20Fractional%20anisotropy%20and,kurtosis%20may%20also%20provide%20greater>`_, `pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/24231848/#:~:text=throughout%20development%2C%20predominantly%20in%20the,birth%20to%204%20years%207>`_).
- **Adult lifespan and aging white matter:** Coutu et al. (2014) found **MK and kurtosis anisotropy** decline with age and that **MK** shows a clearer linear association with advancing age than FA/MD, indicating progressive loss of microstructural complexity in WM.

Methodological Warnings and Caveats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Hansen B, Jespersen SN (2017)

  - Title: Recent Developments in Fast Kurtosis Imaging
  - Journal: Front Phys
  - DOI: `https://doi.org/10.3389/fphy.2017.00040`_
  - Metrics impacted: MK (and DKI metrics generally)
  - Caveats summary: Full DKI needs long acquisitions and can be unstable; proposes faster MK estimation via optimized sampling and constrained fitting to improve clinical feasibility.
  - Google Scholar citations: 53; Citation retrieval date: 11/6/25

- Hui ES, Glenn GR, Helpern JA, Jensen JH (2015)

  - Title: Kurtosis analysis of neural diffusion organization
  - Journal: NeuroImage
  - DOI: `https://doi.org/10.1016/j.neuroimage.2014.11.015`_
  - Metrics impacted: Kurtosis tensor metrics (MK, anisotropic vs isotropic kurtosis)
  - Caveats summary: Overall **MK** conflates multiple microstructural sources. Introduces a decomposition framework (KANDO) to separate anisotropic and isotropic components, warning against over-interpretation of MK/KFA without complementary models.
  - Google Scholar citations: 40; Citation retrieval date: 11/6/25

- Henriques RN, Jespersen SN, Jones DK, Veraart J (2021)

  - Title: Toward more robust and reproducible diffusion kurtosis imaging
  - Journal: Magn Reson Med
  - DOI: `https://doi.org/10.1002/mrm.28730`_
  - Metrics impacted: All DKI-derived metrics
  - Caveats summary: Standard unconstrained DKI fitting can be unreliable; a regularized approach with scalar kurtosis constraints improves fidelity and test–retest reproducibility across MK, AK, RK.
  - Google Scholar citations: 54; Citation retrieval date: 11/6/25

- **Acquisition and fitting constraints:** DKI’s higher-order model (4th-order tensor) requires multiple high–b shells and many directions, leading to long scans and lower SNR (`ajronline.org <https://ajronline.org/doi/10.2214/AJR.13.11365#:~:text=,short%20imaging%20protocol%20is>`_). “Fast kurtosis” strategies can focus on efficient MK estimation (`researchgate.net <https://www.researchgate.net/publication/319860497_Recent_Developments_in_Fast_Kurtosis_Imaging#:~:text=Recent%20Developments%20in%20Fast%20Kurtosis,Abstract%20and%20Figures>`_; `scholar.google.com <https://scholar.google.com/citations?user=zxKbj0MAAAAJ&hl=en#:~:text=%E2%80%AABrian%20Hansen%E2%80%AC%20,biomarkers%20from%20fast%20protocols>`_).
- **Interpretation pitfalls (lack of specificity):** DKI metrics are **not tissue-specific**; MK aggregates different sources (density, dispersion, heterogeneity). Hui et al. caution that the “main caveat of DKI is that different kurtosis sources are all conflated” (`discovery.ucl.ac.uk <https://discovery.ucl.ac.uk/id/eprint/10185792/1/1-s2.0-S1053811921011046-main.pdf#:~:text=The%20main%20caveat%20of%20DKI,mono%02exponential>`_).
- **Noise, artifacts, and reproducibility:** Unconstrained DKI can yield non-physical or variable estimates; regularized estimation with plausible bounds improves reproducibility (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/33829542/#:~:text=reproducibility%20of%20the%20kurtosis%20metrics,with%20enhanced%20quality%20and%20contrast>`_).

Glossary
~~~~~~~~

- **DKI** – *Diffusion Kurtosis Imaging*, an MRI technique extending DTI to quantify the non-Gaussian diffusion of water in tissue (`pubmed.ncbi.nlm.nih.gov <https://pubmed.ncbi.nlm.nih.gov/20632416/#:~:text=Quantification%20of%20non,This%20review%20discusses>`_).
- **dki_mk** – *Mean Kurtosis*, the average diffusional kurtosis over all diffusion directions.
- **dki_ak** – *Axial Kurtosis*, kurtosis measured along the principal diffusion direction.
- **dki_rk** – *Radial Kurtosis*, kurtosis measured perpendicular to the principal fiber direction.
- **dki_md** – *Mean Diffusivity* (from DKI fit), analogous to DTI-MD, estimated alongside kurtosis.
- **dki_fa** – *Fractional Anisotropy* (from the diffusion tensor in a DKI dataset).
- **dki_kfa** – *Kurtosis Fractional Anisotropy*, anisotropy of the kurtosis tensor (`onlinelibrary.wiley.com <https://onlinelibrary.wiley.com/doi/10.1002/mrm.22932#:~:text=Estimation%20of%20tensors%20and%20tensor,>`_).
- **dki_ad** – *Axial Diffusivity* (DKI), diffusivity along the dominant fiber direction.
- **dki_rd** – *Radial Diffusivity* (DKI), diffusivity perpendicular to the main fiber direction.
- **dki_linearity**, **dki_planarity**, **dki_sphericity** – Tensor shape metrics (by analogy to diffusion tensor Westin metrics).
- **dkimicro_ad** – *Axial Diffusivity (Intra-axonal)*, diffusivity along axons in the intra-axonal space.
- **dkimicro_ade** – *Axial Diffusivity (Extra-axonal)*, diffusivity parallel to fibers in the extra-axonal space.
- **dkimicro_awf** – *Axonal Water Fraction*, DKI-derived estimate of axon density (`pmc.ncbi.nlm.nih.gov <https://pmc.ncbi.nlm.nih.gov/articles/PMC3136876/#:~:text=meaningful%20interpretation%20of%20DKI%20metrics,be%20determined%20directly%20from%20the>`_).
- **dkimicro_axonald** – *Axonal Diffusivity*, intrinsic diffusivity inside axons (sometimes overlapping with dkimicro_ad).
- **dkimicro_kfa** – *Kurtosis FA of Microstructure*, FA of kurtosis attributable to microstructural factors.
- **dkimicro_md** – *Mean Diffusivity (Microstructural model)*, from intra- and extra-axonal tensors.
- **dkimicro_rd** – *Radial Diffusivity (Intra-axonal)*, radial diffusivity within the axonal compartment.
- **dkimicro_rde** – *Radial Diffusivity (Extra-axonal)*, perpendicular diffusivity in the extra-axonal space (`pmc.ncbi.nlm.nih.gov <https://pmc.ncbi.nlm.nih.gov/articles/PMC3136876/#:~:text=be%20anisotropic%20Gaussian%20and%20characterized,water%20fraction%20obtained%20from%20standard>`_).
- **dkimicro_tortuosity** – *Extra-axonal Tortuosity*, ratio between extra-axonal parallel and perpendicular diffusivities.
- **dkimicro_trace** – *Trace of the Diffusion Tensor (Micro)*, sum of compartment tensor diffusivities weighted by volume fractions.

Uncertainties
~~~~~~~~~~~~~

- *Google Scholar citation counts* for some older studies (Falangola et al. 2008; Paydar et al. 2014) may vary; exact counts above reflect retrieval on 11/6/25.
- Definitions of certain “dkimicro_” metrics involve model-specific assumptions. For example, **dkimicro_ad** vs **dkimicro_axonald** can overlap in meaning depending on the chosen model.

Citations
~~~~~~~~~

- Diffusional kurtosis imaging: the quantification of non-Gaussian water diffusion by means of MRI – PubMed: `https://pubmed.ncbi.nlm.nih.gov/15906300/`_
- Three-dimensional characterization of non-Gaussian water diffusion in humans using diffusion kurtosis imaging – PubMed: `https://pubmed.ncbi.nlm.nih.gov/16521095/`_
- MRI quantification of non-Gaussian water diffusion by kurtosis analysis – PubMed: `https://pubmed.ncbi.nlm.nih.gov/20632416/`_
- Estimation of tensors and tensor-derived measures in diffusional kurtosis imaging – PubMed: `https://pubmed.ncbi.nlm.nih.gov/21337412/`_
- White Matter Characterization with Diffusional Kurtosis Imaging – PMC: `https://pmc.ncbi.nlm.nih.gov/articles/PMC3136876/`_
- Age-related non-Gaussian diffusion patterns in the prefrontal brain – PubMed: `https://pubmed.ncbi.nlm.nih.gov/19025941/`_
- Diffusional Kurtosis Imaging of the Developing Brain – PubMed: `https://pubmed.ncbi.nlm.nih.gov/24231848/`_
- Diffusion Kurtosis Imaging: An Emerging Technique for Evaluating... – AJR: `https://ajronline.org/doi/10.2214/AJR.13.11365`_
- Recent Developments in Fast Kurtosis Imaging – ResearchGate: `https://www.researchgate.net/publication/319860497_Recent_Developments_in_Fast_Kurtosis_Imaging`_
- Brian Hansen – Google Scholar: `https://scholar.google.com/citations?user=zxKbj0MAAAAJ&hl=en`_
- Correlation Tensor MRI deciphers underlying kurtosis sources in stroke – UCL Discovery: `https://discovery.ucl.ac.uk/id/eprint/10185792/1/1-s2.0-S1053811921011046-main.pdf`_
- Toward more robust and reproducible diffusion kurtosis imaging – PubMed: `https://pubmed.ncbi.nlm.nih.gov/33829542/`_
- Erratum: Estimation of tensors and tensor-derived measures in diffusional kurtosis imaging – Wiley: `https://onlinelibrary.wiley.com/doi/10.1002/mrm.22932`_


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