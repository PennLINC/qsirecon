.. include:: ../links.rst

##########################################################
Neurite Orientation Dispersion and Density Imaging (NODDI)
##########################################################


NODDI Foundational Papers
~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-Compartment Modeling**:
NODDI is a three-compartment diffusion MRI model that separates intra-neurite, extra-neurite, and CSF water signals.
This yields more specific microstructural indices than DTI -
notably the intracellular volume fraction (ICVF) as a proxy for neurite density,
and the orientation dispersion index (ODI) for neurite orientation variability :cite:p:`noddi`. 
These metrics disentangle factors contributing to diffusion anisotropy that were conflated in :ref:`DTI <dti_model>` measures like FA :cite:p:`noddi`.

**Estimation Speed**:
A major practical advance was the AMICO algorithm (Daducci et al. 2015), 
which reformulated NODDI fitting as a linear inverse problem. 
AMICO achieves a 1000× speedup in fitting time with minimal loss of accuracy :cite:p:`amico`. 
This enabled large studies and clinical workflows to include NODDI analyses, cementing NODDI's popularity. 
Notably, AMICO does not alter NODDI's metrics; it accelerates their computation.

**Partial Volume Correction**: 
A known issue is that CSF partial-volume can lead to underestimation of neurite density in voxels near ventricles or cortex. 
In 2021, :cite:t:`parker2021not` introduced tissue-fraction-modulated ICVF and ODI, 
scaling NODDI metrics by the tissue signal fraction (1 - ISOVF) :cite:p:`Zhao2021BrainDevelopment`. 
This adjustment was shown to remove artifactual group differences that were driven by differing CSF contamination rather than true tissue changes :cite:p:`parker2021not`. 
Tissue modulated maps are produced by default in QSIRecon.


NODDI Studies Across the Lifespan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Early Development (Infancy)**: 
NODDI studies in infants show dramatic microstructural maturation within the first years of life. 
For example, 
neurite density (ICVF) in major tracts increases steeply from birth to age 3 :cite:p:`Jelescu2015WMTINODDI`, 
reflecting rapid axonal growth and myelination. 
This is accompanied by increasing restriction of diffusion in the extra-cellular space 
(consistent with growing fibers and tighter packing).

**Childhood and Adolescence**: 
Neurite density continues to rise through childhood and adolescence, though the rate slows with age. 
Multiple studies (e.g. :cite:t:`Genc2017NDIAgeDevelopment`, :cite:t:`Mah2017NODDIChildDevelopment`) 
found strong positive correlations between age and ICVF/NDI in children.
By contrast, orientation dispersion (ODI) remains relatively stable during childhood,
indicating that fiber organization (coherence) does not markedly increase after early childhood in most regions :cite:p:`Mah2017NODDIChildDevelopment`.
The net effect is that white matter FA increases in youth are chiefly driven by increasing neurite density (more and thicker axons with more myelin),
rather than fibers becoming straighter or more aligned. 
Children with higher NDI tend to have higher FA even if ODI is unchanged, 
meaning microstructural density is a key determinant of developmental differences :cite:p:`Mah2017NODDIChildDevelopment`.

**Late Adolescence to Early Adulthood**: 
Neurite density in many tracts appears to plateau by the early 20s, 
reaching peak or near-peak values, while ODI might begin to creep upward. 
:cite:t:`Chang2015NODDIDataset` noted that NDI followed a logarithmic growth curve that leveled off in the 20s-30s, 
whereas ODI began an exponential upward trend in the 30s.
This implies that during the transition to adulthood, continued incremental myelination is eventually counteracted by emerging microscopic disorganization 
(potentially reflecting early branching/pruning or cumulative minor damage), 
which foreshadows the patterns seen in later aging.

**Mid-Life and Healthy Aging**: 
In mid-to-late adulthood, NODDI reveals progressive loss of neurite density and increasing fiber dispersion. 
Large-scale data from the UK Biobank showed that older age is associated with lower ICVF 
and higher ODI across virtually all white matter tracts :cite:p:`Lawrence2021UKBiobankWhiteMatter`. 
Notably, these microstructural changes can be detected even in healthy adults with no disease, reflecting normative brain aging. 
Age-related ODI increases often accelerate in the later decades, 
consistent with accumulating structural disintegration.
:cite:t:`Lawrence2021UKBiobankWhiteMatter` also found that CSF fraction (ISOVF) 
increases with age, indicating expanding extracellular space.

**Clinical and Cognitive Relevance**: 
The lifespan changes in NODDI metrics align with known windows of brain plasticity and decline. 
The steep NDI increase in childhood corresponds to learning and cognitive development, 
whereas rising ODI in late life correlates with cognitive slowing and increased white matter vulnerability. 
Importantly, NDI has been found to be a better predictor of chronological age than standard DTI measures in youth :cite:p:`Genc2017NDIAgeDevelopment`.


NODDI Methodological Warnings and Caveats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Model Assumptions and Biases**: 
NODDI relies on several fixed model assumption. 
Notably that all intra-axonal and extra-axonal water shares the same diffusion coefficient 
(typically d‖ = 1.7 µm²/ms) 
and that fibers in a voxel have a single average orientation dispersion (Watson distribution). 
In reality, these assumptions are often violated: 
gray matter neurites have slower diffusion, 
pathology can alter compartment diffusivities, 
and multiple fiber populations can exist. 
As a result, NODDI parameter estimates may be biased if these conditions aren't met :cite:p:`Zhao2021BrainDevelopment`. 
For example, 
using the default d‖ in cortical gray matter can lead to overestimation of ICVF (since true diffusion is slower).
Investigators have to adjust this value for different tissues or accept some bias :cite:p:`Zhao2021BrainDevelopment`. 
Simplifying assumptions are necessary to keep NODDI practical, 
but users should understand they introduce systematic errors in certain contexts.


**Degeneracy and Parameter Coupling**: 
A fundamental challenge with multi-compartment models like NODDI is that different parameter combinations can produce very similar diffusion signals. 
There is a trade-off between neurite density and dispersion, 
for instance: 
a voxel with fewer, well-aligned axons can have a similar diffusion profile to one with more axons that are highly dispersed. 
This can lead to degenerate solutions where the fitting algorithm might converge on one of several "equivalent" parameter sets :cite:p:`jelescu2017design`. 
In practice, 
it means NODDI outputs are not always unique, 
especially if data quality (SNR, number of diffusion directions, b-values) is limited. 


**Interpretational Specificity**: 
While NODDI's indices are more directly linked to microstructure than DTI's, they are not one-to-one with histology. 
ICVF, for example, is often called “neurite density index,” 
but it doesn't strictly equal axon count or volume fraction in a straightforward way.
It's influenced by dendrites in gray matter, by glial processes, and by whether axons are myelinated or not. 
ODI is likewise a proxy for fiber orientation dispersion, 
but it can be increased by diverse scenarios
(actual fanning of fibers, beading/swelling of axons, or mixture of fiber orientations). 
Moreover, pathology can confound these metrics: 
e.g., inflammation adds isotropic water which lowers ICVF and raises ISOVF, 
mimicking axonal loss.
NODDI is best used to compare groups or conditions, rather than to obtain exact "neurite counts".


References
~~~~~~~~~~

.. bibliography::
   :style: unsrt
   :filter: cited