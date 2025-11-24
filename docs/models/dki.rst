.. include:: ../links.rst

.. _dki_model:

##################################
Diffusional kurtosis imaging (DKI)
##################################


***********************
DKI Foundational Papers
***********************

**DKI Concept Introduction**:
DKI was first introduced by :cite:t:`jensen2005dki` as an extension of
:ref:`DTI <dti_model>` to measure diffusion non-Gaussianity in tissues.
Their seminal work defined *diffusional kurtosis* as a quantitative marker,
showing that normal white matter has substantially higher kurtosis than gray matter.
Building on this, :cite:t:`lu2006dki` provided the first full mathematical derivation of the kurtosis tensor,
introduced rotational invariants like mean kurtosis (MK),
reported reproducible MK values,
and showed that kurtosis anisotropy can reveal complex fiber geometries.
Jensen and Helpern's 2010 review :cite:p:`jensen2010dki` consolidated the DKI model,
formalizing MK, as well as axial and radial kurtosis
(AK, RK) as rotationally invariant descriptors,
discussed practical acquisition requirements and highlighted DKI's sensitivity to tissue heterogeneity.

**DKI Methodological Improvements**:
:cite:t:`tabesh2011dki` introduced constrained least-squares estimation to ensure physically valid DKI fits,
defined Kurtosis Fractional Anisotropy (KFA),
and provided closed-form formulas for MK and RK.
:cite:t:`fieremans2011dki` extended DKI toward microstructural modeling of white matter,
deriving Axonal Water Fraction (AWF) and extra-axonal tortuosity from DKI data
and aligning DKI-derived parameters with known tissue features.
:cite:t:`henriques2021dki` introduced Mean Signal DKI (MSDKI),
which estimates DKI parameters more robustly.

*******************************
DKI Studies Across The Lifespan
*******************************

**Normal aging:**
:cite:t:`falangola2008dki` showed age-related changes in DKI metrics across the healthy lifespan;
MD increased and FA decreased in the oldest group, while MK exhibited distinct trends across the lifespan.

**Early development:**
:cite:t:`paydar2014dki` demonstrated that FA and MK rise with age in WM,
but MK continues to increase after FA plateaus;
MK also revealed GM maturation undetectable by FA.

**Adult lifespan and aging white matter:**
:cite:t:`coutu2014dki` found that MK and kurtosis anisotropy declines with age
and that MK shows a clearer linear association with advancing age than FA/MD,
indicating progressive loss of microstructural complexity in WM.

***************************************
DKI Methodological Warnings and Caveats
***************************************

**Acquisition and fitting constraints:**
DKI's higher-order model (4th-order tensor)
requires multiple high-b shells and many directions,
leading to longer scans and lower SNR :cite:p:`steven2014dki`.
However most modern multi-shell dMRI scans are compatible with DKI.

**Interpretation pitfalls (lack of specificity):**
DKI metrics are not tissue-specific;
MK aggregates different sources (density, dispersion, heterogeneity).
:cite:t:`alves2022cti` caution that the "main caveat of DKI is that different kurtosis sources are all conflated".

**Noise, artifacts, and reproducibility:**
Unconstrained DKI can yield non-physical or variable estimates;
regularized estimation with plausible bounds improves reproducibility :cite:p:`henriques2021robustdki`.
Further, if data is not adequately denoised and de-Gibbs'ed there will be holes in the maps.
:ref:`MKDKI <msdki_model>` can improve this.


**********
References
**********

.. bibliography::
   :style: unsrt
   :cited:
