   .. include:: links.rst

   .. _pyafq_model:

   ################################################
   pyAFQ (Automated Fiber Quantification in Python) 
   ################################################

   pyAFQ is part of a software ecosystem for brain tractometry processing, analysis,
   and insight :cite:p:`kruper2025`. It is a Python implementation of the original
   MATLAB-based Automated Fiber Quantification (AFQ) pipeline :cite:p:`pyafq2`.
   pyAFQ automates the identification and quantification of white matter fiber tracts
   from diffusion MRI data. It implements a pipeline that includes tractography,
   delineation of tracts based on anatomical constraints, and the extraction of diffusion
   metrics along the length of the tracts for the purpose of tractometry. pyAFQ is designed
   to be flexible and extensible, allowing researchers to customize analyses and integrate
   with other neuroimaging software, including AFQ-Insight :cite:p:`richiehalford2021`, 
   a Python library for statistical learning of tractometry data.

   Detailed documentation can be found at
   `pyAFQ Documentation <https://tractometry.org/pyAFQ/index.html>`.


   ******************
   pyAFQ in QSIRecon
   ******************

   Tractometry with pyAFQ is supported in QSIRecon through the pyAFQ package 
   (see :func:`~qsirecon.workflows.recon.pyafq.init_pyafq_wf`).

   The pyAFQ approach is accessible in a reconstruction specification by using a node with
   ``action: pyafq_tractometry`` and ``software: pyAFQ``.
   Also see :class:`qsirecon.interfaces.pyafq.PyAFQRecon`.

   In QSIRecon, pyAFQ can accept as input tractography or maps of tissue properties that 
   were generated with other pipelines. This flexibility allows researchers to combine 
   preferred tractography and reconstruction methods with standardized tract identification 
   and quantification.

   *************************
   pyAFQ Foundational Papers
   *************************

   Tractometry is an approach to analyzing diffusion MRI data that quantifies tissue 
   properties along the trajectory of white matter pathways. Rather than summarizing 
   measurements across an entire fiber bundle with a single mean value, tractometry 
   generates profiles that capture how diffusion metrics vary along the length of a 
   tract. Early approaches to tract-specific analysis include TRACULA (TRActs 
   Constrained by UnderLying Anatomy) :cite:p:`tracula` and PASTA (Pointwise assessment 
   of streamline tractography attributes) :cite:p:`pasta`.

   The Automated Fiber Quantification (AFQ) method :cite:p:`pyafq` automated 
   tractometry by standardizing tract identification based on waypoint ROIs and
   generating quantitative profiles by resampling each streamline within a tract into 
   100 evenly spaced nodes. A parameter value vector is then produced along the tract, 
   with parameters weighted by their distance to the core of each fiber bundle. 
   The AFQ method established several key findings: first, while FA values vary 
   substantially within a tract, the resulting Tract FA Profile structure remains 
   highly consistent across healthy subjects. Second, developmental changes in FA 
   occur not along the entire tract uniformly, but rather are concentrated at 
   specific positional segments.

   *********************************
   pyAFQ Studies Across The Lifespan
   *********************************

   pyAFQ has been validated and applied across diverse age ranges and populations in
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

   **********
   References
   **********

   .. bibliography::
      :style: unsrt
      :filter: docname in docnames