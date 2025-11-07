.. include:: links.rst

####################################
Generalized q-Sampling Imaging (GQI)
####################################

GQI is a model-free approach that estimates water
diffusion orientation distribution functions (dODFs) using an analytic transform of
the diffusion signal :cite:p:`yeh2010`.
Like the other models, GQI produces a number
of parametric microstructure maps such as generalized fractional anisotropy (GFA),
quantitative anisotropy (QA), and isotropic component (ISO) :cite:p:`yeh2013`.


GQI Foundational Papers
~~~~~~~~~~~~~~~~~~~~~~~

- Generalized q-Sampling Imaging (GQI) was introduced as an analytical, model-free diffusion MRI reconstruction method that bridges Q-ball imaging (single-shell HARDI) and diffusion spectrum imaging (grid sampling) :cite:p:`yeh2010`. It estimates the spin distribution function (SDF), an ODF-like representation of diffusion directions, from a variety of sampling schemes.
- Generalized Fractional Anisotropy (GFA) – originally defined in QBI :cite:p:`yeh2010` – is an ODF-derived analog of tensor FA. It quantifies the angular variance of diffusion and is output by GQI for each voxel. GFA provides a unitless 0–1 measure of anisotropy but, like FA, can be confounded by complex fiber architecture (e.g. crossing fibers can lower GFA).
- Quantitative Anisotropy (QA) was a key new metric introduced with GQI :cite:p:`yeh2010`. Unlike FA/GFA which are normalized measures per voxel, QA measures the absolute density of diffusional spins along a specific fiber orientation (per fiber population). By subtracting the isotropic background signal, QA directly reflects the relative volume fraction or “strength” of each fiber pathway in the voxel. This makes QA more specific to fiber integrity and less sensitive to isotropic partial volumes.
- Isotropic Diffusion (ISO) refers to the isotropic component separated out by GQI. In practice, GQI estimates the background isotropic diffusion in a voxel (often approximated by the minimum SDF value) :cite:p:`yeh2010`. This yields an ISO value that quantifies free-water diffusion (e.g. CSF or edema) independently. GQI’s ability to isolate isotropic diffusion helps reduce free-water contamination in anisotropy measures.
- Foundational studies validated GQI and its metrics: :cite:t:`yeh2010` showed GQI’s ODF reconstruction accuracy comparable to DSI/QBI and introduced QA. Subsequent work :cite:p:`yeh2013deterministic` demonstrated QA’s practical advantages – it remains stable despite crossing fibers or edema, leading to improved tractography with fewer false positives. Early applications in development :cite:p:`lim2015rabbitbrain` highlighted that GQI metrics like GFA and QA can track brain maturation, revealing age-related increases in anisotropy even in regions where DTI failed to detect changes (e.g. hippocampal development).
- Takeaway: GQI extends diffusion MRI beyond the tensor model, providing directional resolution in complex fiber regions and introducing metrics (QA, ISO, GFA) that offer richer microstructural information. General neuroimaging users should understand that QA and GQI-derived anisotropies allow fiber-specific analysis (robust to crossing and free water), while GFA generalizes the concept of FA to ODFs. These innovations laid the groundwork for more accurate mapping of white-matter architecture in research and, potentially, clinical imaging.

GQI Studies Across The Lifespan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Normal aging:** :cite:t:`falangola2008dki` showed age-related changes in DKI metrics across the healthy lifespan; MD increased and FA decreased in the oldest group, while **MK** exhibited distinct trends across the lifespan.
- **Early development:** :cite:t:`paydar2014dki` demonstrated that **FA and MK rise with age in WM**, but MK continues to increase after FA plateaus; MK also revealed GM maturation undetectable by FA.
- **Adult lifespan and aging white matter:** :cite:t:`coutu2014dki` found **MK and kurtosis anisotropy** decline with age and that **MK** shows a clearer linear association with advancing age than FA/MD, indicating progressive loss of microstructural complexity in WM.

GQI Methodological Warnings and Caveats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**********
References
**********

.. bibliography::
   :style: unsrt
   :filter: cited