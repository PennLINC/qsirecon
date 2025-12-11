.. include:: links.rst

.. _pyafq_model:

#############################################
pyAFQ (Automated Fiber Quantification in Python)
#############################################

pyAFQ is part of a software ecosystem for brain tractometry processing, analysis,
and insight :cite:p:`kruper2025`. It is a Python implementation of the original
MATLAB-based Automated Fiber Quantification (AFQ) pipeline :cite:p:`pyafq2`.
pyAFQ automates the identification and quantification of white matter fiber tracts
from diffusion MRI data. It implements a pipeline that includes tractography,
delineation of tracts based on anatomical constraints, and the extraction of diffusion metrics along the length of the tracts for the purpose of tractometry.
pyAFQ is designed to be flexible and extensible, allowing researchers to
customize analyses and integrate with other neuroimaging software. For example, in `qsirecon` it can accept as input tractography or maps of tissue properties that were generated with other pipelines. It also natively supports
various diffusion models and metrics, facilitating comprehensive studies of white
matter microstructure and connectivity. The software is actively maintained
and widely used in the neuroimaging community, with contributions from multiple 
research groups and institutions supporting the development and validation of
pyAFQ :cite:p:`pyafq`.

Detailed documentation can be found at
`pyAFQ Documentation <https://tractometry.org/pyAFQ/index.html>`.


pyAFQ Foundational Papers
~~~~~~~~~~~~~~~~~~~~~~~

The field of neuroimaging, particularly dMRI analysis, has historically faced
methodological challenges regarding the reproducibility and robustness of its results.
To address these inherent concerns and establish the trustworthiness of computational
pipelines, the focus has shifted toward formalized quantitative analysis. This
systematic approach is defined by tractometry, an analysis technique that moves beyond
simple visualization (tractography) to quantitatively assess the physical properties
of white matter pathways derived from dMRI scalars.

The methodological core of pyAFQ is rooted in the concept of "Tract Profiles"
introduced in the seminal work by Yeatman et al. in 2012. This innovation addressed
a significant limitation in traditional dMRI analysis: relying solely on mean
diffusion measures across an entire tract obscures critical, systematic variability
in tissue properties along the tract’s length. This along-tract variability arises
naturally, as different populations of axons enter and exit the fiber bundle, and,
critically, because disease or injury may strike at specific, localized positions
along the tract.   

The Tract Profile methodology involves generating a high-resolution, one-dimensional
profile of quantitative metrics (such as Fractional Anisotropy, FA) along the
trajectory of the fiber bundle. This is achieved by resampling each streamline
within a tract into 100 evenly spaced nodes. A parameter value vector is then produced
along the tract, with parameters weighted by their distance to the core of each fiber
bundle.   

This along-tract quantification provides a mechanism for significantly increased
statistical power and biological specificity compared to traditional global 
measurements. The original research established several key findings supporting this 
high-resolution approach: first, while FA values vary substantially within a tract, 
the resulting Tract FA Profile structure remains highly consistent across healthy 
subjects. Second, and most importantly for lifespan studies, developmental changes in 
FA occur not along the entire tract uniformly, but rather are concentrated at specific
positional segments.

pyAFQ Studies Across The Lifespan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pyAFQ has been validated and applied across diverse age ranges and populations, and in
cross-sectional and longitudinal study designs, demonstrating its robustness for 
studying white matter development from infancy through adulthood. For example, pyAFQ was
sensitive to developmental changes in white matter along two fundamental axes - a
deep-to-superficial gradient where superficial tract regions near the cortical surface
show greater age-related change, and alignment with the sensorimotor-association 
cortical hierarchy where tract ends adjacent to sensorimotor cortices mature earlier 
than those near association cortices :cite:p:`luo2025`. 

babyAFQ :cite:p:`grotheer2022` addresses the unique challenges of tractography in infant 
diffusion MRI by providing specialized waypoint ROIs and bundle definitions using the
UNC Neonatal template. In a longitudinal study of infants at 0, 3, and 6 months, 
babyAFQ was used to quantify R1 development (a myelin-sensitive measure) across white 
matter bundles, revealing nonuniform development with faster maturation in less mature 
bundles and along inferior-to-superior and anterior-to-posterior spatial gradients. 

pyAFQ has also been used in studies which examined how white matter properties change 
in response to intensive reading interventions in both 
preschoolers :cite:p:`caffarra2025` and school-aged children :cite:p:`yablonski2025` 
with reading disabilities, providing insights into both the rapid plasticity of white 
matter pathways and the neural mechanisms underlying reading improvement.

pyAFQ is compatible with other open-source neuroimaging tools, and has been used in 
large-scale neuroimaging studies such as the Human Connectome 
Project (HCP) :cite:p:`kruper2024` and Healthy Brain Network (HBN) among others.

It is also part of a larger ecosystem, including AFQ-Insight :cite:p:`richiehalford2021`,
a Python library for statistical learning of tractometry data.

**********
References
**********

.. bibliography::
   :style: unsrt
   :filter: docname in docnames