.. include:: ../links.rst

####################################
Generalized q-Sampling Imaging (GQI)
####################################

GQI is a model-free approach that estimates water
diffusion orientation distribution functions (dODFs) using an analytic transform of
the diffusion signal :cite:p:`yeh2010`.
Like the other methods, GQI produces a number
of parametric microstructure maps such as generalized fractional anisotropy (GFA),
quantitative anisotropy (QA), and isotropic component (ISO) :cite:p:`yeh2013`.


GQI Foundational Papers
~~~~~~~~~~~~~~~~~~~~~~~

- Generalized q-Sampling Imaging (GQI) was introduced as an analytical,
  model-free diffusion MRI reconstruction method that bridges Q-ball imaging (single-shell HARDI)
  and diffusion spectrum imaging (grid sampling) :cite:p:`yeh2010`.
  It estimates the spin distribution function (SDF),
  an ODF-like representation of diffusion directions,
  from a variety of sampling schemes.
- Generalized Fractional Anisotropy (GFA),
  originally defined in QBI :cite:p:`tuch2004q`,
  is an SDF-derived analog of the DTI-based Fractional Anisotropy (FA).
  It quantifies the angular variance of diffusion and is output by GQI for each voxel. 
  GFA provides a unitless 0-1 measure of anisotropy but, like FA, 
  can be confounded by complex fiber architecture (e.g., crossing fibers can lower GFA).
- Quantitative Anisotropy (QA) was a key new metric introduced with GQI :cite:p:`yeh2010`. 
  Unlike FA/GFA which are normalized measures per voxel, 
  QA measures the absolute density of diffusional spins along a specific 
  SDF lobe.
  SDF lobes are interpreted as fiber orientations (fixels in MRtrix terms).
  By subtracting the isotropic background signal, 
  QA directly reflects the relative volume fraction or "strength" of each fixel in the SDF. 
- Isotropic Diffusion (ISO) refers to the isotropic component separated out by GQI. 
  In practice, GQI estimates the background isotropic diffusion in a voxel 
  as the minimum SDF value :cite:p:`yeh2010`. 
- Foundational studies validated GQI and its metrics: :cite:t:`yeh2010` 
  showed GQI's ODF reconstruction accuracy comparable to DSI/QBI and introduced QA. 
  Subsequent work :cite:p:`yeh2013` demonstrated QA's practical advantages:
  it remains stable despite crossing fibers or edema, 
  leading to improved tractography with fewer false positives.


GQI Studies Across The Lifespan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GQI has predominantly been used as the basis for tractography.
There are, however some studies looking at GQI-based metrics across the lifespan.

- Development (animal model): GQI-derived metrics generally increase during brain maturation, 
  reflecting strengthening and organization of white matter. 
  For example, :cite:t:`lim2015rabbitbrain` showed that rabbit brains from infancy to adulthood had rising GFA
  and normalized quantitative anisotropy (NQA) values in major tracts, 
  mirroring the well-known FA increases in development. 
  Importantly, 
  GQI metrics unveiled maturation in regions like the hippocampus (gray matter) that DTI FA did not capture,
  indicating GQI's potential to detect subtler microstructural ordering.
- Pathological Aging and Injury (animal model): 
  GQI metrics have been applied to models of aging-related brain injury (e.g., radiation exposure in older rabbits). 
  :cite:t:`shen2015rabbitbraininjury` found QA and ISO could track complex temporal changes 
  (inflammation, edema, recovery) post-radiation that weren't fully captured by FA. 
  Notably, ISO (isotropic diffusion) tends to increase in situations of edema or tissue loss
  which is often more pronounced in aging brains (e.g. ventricular expansion, lesions). 
  GQI explicitly quantifies this isotropic water component, 
  which can help disentangle true axonal changes (QA) from mere fluid-related changes (ISO) in elderly or disease populations.


GQI Methodological Warnings and Caveats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Data Requirements: GQI's accuracy and outputs depend on adequate diffusion sampling. 
  In practice, users should ensure high angular resolution or multi-shell data for GQI.
- Reproducibility and Normalization:
  Because QA is an absolute signal metric, 
  it can vary with scanner hardware and protocol.
  Between scanners or sessions, 
  differences in coil gains, b-value, or SNR could lead to QA changes unrelated to biology. 
  This is a concern especially in multi-center studies or longitudinal studies with system upgrades.
- Model Limitations vs. DTI: 
  GQI expands capability at the cost of complexity. 
  Unlike the simple tensor, GQI has a parameter (the length ratio ~1.2-1.3) 
  to tune the sensitivity to diffusion distance.
  Using a wrong value could under- or over-estimate QA and fiber (fixel) counts.
- Clinical Translation Caveat: 
  While GQI metrics (QA, GFA, ISO) offer richer information, 
  their biological specificity is still under investigation.


References
~~~~~~~~~~

.. bibliography::
   :style: unsrt
   :filter: cited