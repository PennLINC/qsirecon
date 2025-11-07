.. include:: links.rst

################################
Diffusion kurtosis imaging (DKI)
################################

Water diffusion in the brain is affected by the physical structures that make up neurons and organelles.
Instead of freely diffusing through space, water encounters barriers from myelin, cell membranes and other structures that introduce non-Gaussian features into the water diffusion distribution.
The Diffusion Kurtosis Imaging (DKI; :cite:p:`jensen2005dki`) model extends the of the DTI model by adding an additional 15 parameters that capture the deviations from Gaussianity missed when fitting the simple 6 parameter DTI model.
The DKI model incorporates data from all shells, potentially estimating the same scalar maps from DTI (FA, MD, etc) more accurately than a traditional tensor fit :cite:p:`henriques2021dki`.
In addition to the measures from DTI, the DKI model also allows one to compute additional scalars derived from the kurtosis tensor such as mean kurtosis (MK), radial kurtosis (RK), and axial kurtosis (AK) :cite:p:`jensen2010dki`.
DKI’s sensitivity to non-Gaussian diffusion makes it useful for capturing the interaction of water with more complex tissue features.

DKI Foundational Papers
~~~~~~~~~~~~~~~~~~~~~~~

- DKI was first introduced by :cite:t:`jensen2005dki` as an extension of DTI to measure **diffusion non-Gaussianity** in tissues. Their seminal work defined **diffusional kurtosis** as a quantitative marker, showing that normal white matter has substantially higher kurtosis than gray matter.
- Building on this, :cite:t:`lu2006dki` provided the first full **mathematical derivation of the kurtosis tensor**, introduced rotational invariants like **mean kurtosis (MK)**, reported reproducible MK values, and showed that kurtosis anisotropy can reveal complex fiber geometries.
- Jensen and Helpern's 2010 review :cite:p:`jensen2010dki` consolidated the DKI model, formalizing **MK, AK, RK** as rotationally invariant descriptors, discussed practical acquisition requirements and highlighted DKI’s sensitivity to tissue heterogeneity.
- :cite:t:`tabesh2011dki` introduced **constrained least-squares estimation** to ensure physically valid DKI fits, defined **KFA**, and provided closed-form formulas for MK and RK.
- :cite:t:`fieremans2011dki` extended DKI toward **microstructural modeling of white matter**, deriving **AWF** and **extra-axonal tortuosity** from DKI data and aligning DKI-derived parameters with known tissue features.

DKI Studies Across The Lifespan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Normal aging:** :cite:t:`falangola2008dki` showed age-related changes in DKI metrics across the healthy lifespan; MD increased and FA decreased in the oldest group, while **MK** exhibited distinct trends across the lifespan.
- **Early development:** :cite:t:`paydar2014dki` demonstrated that **FA and MK rise with age in WM**, but MK continues to increase after FA plateaus; MK also revealed GM maturation undetectable by FA.
- **Adult lifespan and aging white matter:** :cite:t:`coutu2014dki` found **MK and kurtosis anisotropy** decline with age and that **MK** shows a clearer linear association with advancing age than FA/MD, indicating progressive loss of microstructural complexity in WM.

DKI Methodological Warnings and Caveats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Acquisition and fitting constraints:** DKI’s higher-order model (4th-order tensor) requires multiple high–b shells and many directions, leading to long scans and lower SNR :cite:p:`steven2014dki`. “Fast kurtosis” strategies can focus on efficient MK estimation :cite:p:`hansen2017fastkurtosis`.
- **Interpretation pitfalls (lack of specificity):** DKI metrics are **not tissue-specific**; MK aggregates different sources (density, dispersion, heterogeneity). Hui et al. caution that the “main caveat of DKI is that different kurtosis sources are all conflated” :cite:p:`alves2022cti`.
- **Noise, artifacts, and reproducibility:** Unconstrained DKI can yield non-physical or variable estimates; regularized estimation with plausible bounds improves reproducibility :cite:p:`henriques2021robustdki`.


**********
References
**********

.. bibliography::
   :style: unsrt
   :filter: cited