.. include:: ../links.rst

.. _mapmri_model:

######################################
Mean Apparent Propagator MRI (MAP-MRI)
######################################


*******************
MAP-MRI in QSIRecon
*******************

MAPMRI reconstruction is supported in QSIRecon using the
DIPY package (see :func:`~qsirecon.workflows.recon.dipy.init_dipy_mapmri_recon_wf`)
or the TORTOISE package (see :func:`~qsirecon.workflows.recon.tortoise.init_tortoise_estimator_wf`).

The DIPY approach is accessible in a reconstruction specification by using a node with
``action: MAPMRI_reconstruction`` and ``software: Dipy``.
Also see :class:`qsirecon.interfaces.dipy.MAPMRIReconstruction`.

The TORTOISE approach is accessible in a reconstruction specification by using a node with
``action: estimate``, ``software: TORTOISE``, and the ``parameters: estimate_mapmri`` subdictionary.
Also see :class:`qsirecon.interfaces.tortoise.EstimateMAPMRI`.


***************************
MAP-MRI Foundational Papers
***************************

**MAP-MRI Framework and Novel Diffusion Metrics**:
:cite:t:`ozarslan2013` introduced MAP-MRI as a comprehensive 3D
q-space model that fits the diffusion signal with Hermite basis functions,
enabling the direct computation of the ensemble average propagator in voxel-wise
fashion :cite:p:`ozarslan2013`. This foundational work defined new scalar indices
derived from the propagator, including RTOP (Return to origin probability),
RTAP (Return to axis probability), RTPP (Return to plane probability),
propagator anisotropy (PA), and non-Gaussianity (NG) – which quantify diffusion
restrictions and anisotropy beyond the capabilities of DTI.

The return-to-origin probability (RTOP) was defined as the probability of zero
displacement, sensitive to cell density and restrictions, while RTAP/RTPP
measure probabilities of displacements confined to an axis or plane
:cite:p:`ozarslan2013`. :cite:t:`ozarslan2013` also formulated propagator
anisotropy (PA) as a generalized anisotropy index reflecting the deviation of
the full propagator from an isotropic counterpart. A global non-Gaussianity (NG)
index was introduced to capture the total deviation of the diffusion propagator
from a Gaussian form, reflecting microstructural complexity not captured by a
single diffusion tensor :cite:p:`ozarslan2013`.

**Clinical Feasibility**: :cite:t:`avram2016` demonstrated that MAP-MRI acquisition
and reconstruction are feasible in vivo within clinically practical scan times
(~10 minutes). This study confirmed that MAP-MRI metrics produce consistent
anatomical contrast and added microstructural information compared to DTI.
Notably, :cite:t:`avram2016` introduced direction-specific NG metrics – splitting
non-Gaussianity into parallel and perpendicular components – analogous to axial
and radial kurtosis. These refinements allowed more nuanced characterization
of diffusion complexity along and across fiber directions, revealing,
for example, that MAP-derived PA is less confounded by fiber crossings
than DTI's FA :cite:p:`avram2016`.

**Laplacian Regularization (MAPL)**: :cite:t:`fick2016` addressed a key
methodological caveat of the original MAP-MRI – its potential to overfit
noisy or sparse data due to many free parameters. They introduced MAPL,
a Laplacian norm regularization of the MAP-MRI coefficients. This
regularization imposes smoothness on the fitted propagator, significantly
reducing spurious oscillations and unstable estimates. :cite:t:`fick2016` showed
that MAPL outperformed the unregularized MAP-MRI and other basis expansions in
phantoms and accelerated acquisitions, enabling reliable estimation of
propagator metrics with fewer diffusion sampling points. The MAPL approach
also improved the downstream reliability of biophysical model fits (e.g.,
axon diameter from AxCaliber, neurite density from NODDI) by using
the regularized MAP signal as a starting point.

**Derivatives and Extensions**: Collectively, these foundational works established
the MAP-MRI model and its key derivative metrics. They set the stage for MAP-MRI's
adoption in advanced neuroimaging by defining how to obtain scalar maps of mean
squared displacement (MSD) and q-space inverse variance (QIV, measures of diffusion
dispersion), Laplacian norm (quantifying propagator roughness for regularization),
and others directly from the fitted propagator. The introduction of regularization
and directional metrics addressed early limitations, making MAP-MRI more robust
and interpretable for general use.

***********************************
MAP-MRI Studies Across The Lifespan
***********************************

**White Matter Maturation vs. Aging**:
In white matter, MAP-MRI indices have highlighted distinct patterns of change.
Propagator anisotropy (PA) –
a microscale analog of anisotropy – typically peaks in early-to-mid adulthood
and then decreases with advanced age, indicating a loss of fiber organization
and coherence in aging white matter :cite:p:`bouhrara2023`. Conversely, the non-Gaussianity (NG) of
diffusion tends to be low in youth (when diffusion is relatively restricted and
orderly) and rises in older adults, consistent with increased diffusion
heterogeneity, extra-cellular water content, and tissue complexity in aging
brains. Notably, :cite:t:`bouhrara2023` found significantly higher NG alongside
lower PA in older white matter compared to younger adults, reflecting
microstructural degeneration such as demyelination and axonal packing
loss in senescence.

**Gray Matter Findings**: In cortical and deep gray matter, diffusion is more
isotropic, so MAP-MRI changes are more subtle but still informative. Lifespan
studies have reported modest declines in gray matter propagator anisotropy with
age and slight increases in NG, though much less dramatic than in white matter.
These trends may correspond to age-related dendritic pruning or iron deposition.
The 2023 study by Bouhrara et al. noted detectable MAP-MRI changes in cortex
across ages, but the contrast between young and old was most pronounced in
white matter metrics, indicating that white matter microstructure undergoes
more significant age-related remodeling :cite:p:`bouhrara2023`.


*******************************************
MAP-MRI Methodological Warnings and Caveats
*******************************************

**Data Requirements and Sampling Bias**: :cite:t:`muftuler2021` demonstrated that
diffusion propagator metrics, including those derived from MAP-MRI, can be biased
when simultaneous multi-slice (SMS) acceleration is used during acquisition.
The study found that residual signal leakage between simultaneously excited slices
in SMS acquisitions can introduce systematic errors in diffusion model parameter
estimation, leading to biased propagator metrics. This finding is particularly
relevant for MAP-MRI studies, as the method relies on accurate q-space sampling
to reconstruct the diffusion propagator. Researchers should be aware that SMS
acceleration, while reducing scan time, may compromise the accuracy of MAP-MRI
derived metrics such as RTOP, RTAP, RTPP, PA, and NG. Careful consideration of
acquisition parameters and potential correction methods is warranted when using
SMS acceleration with MAP-MRI.

**Noise Sensitivity and Regularization**: Because MAP-MRI involves fitting many
coefficients, it is inherently more sensitive to noise than simpler models.
Without constraints, the fitting can produce oscillations or non-physical
propagator values (including negative probabilities). The introduction of the
positivity constraint and Laplacian regularization (MAPL) was specifically to
combat this: these techniques greatly stabilize the fits in noisy or
undersampled data, improving the reliability of metrics :cite:p:`fick2016`.
Users should be aware that regularization is typically necessary for MAP-MRI
– unregularized fits might overfit noise, yielding erratic metric maps.
Tuning of the regularization weight (e.g. via cross-validation as suggested
by :cite:t:`fick2016`) or using the built-in methods in software (like DIPY's
GCV for MAPL) is recommended for robust results :cite:p:`fick2016` (see also
`DIPY documentation <https://docs.dipy.org/stable/examples_built/reconstruction/reconst_mapmri.html>`_).

**Uncertainty and Reproducibility**: A key caveat is the uncertainty in
MAP-MRI metric estimates. :cite:t:`gu2019` demonstrated that the variability
(standard deviation) of MAP-derived metrics is non-negligible and depends on
factors like how many diffusion directions and shells were acquired.
For instance, RTOP or PA values in a given voxel can fluctuate due to noise by
a comparable magnitude to group differences if the data quality is low.
This implies that statistical confidence intervals or bootstrap methods should
accompany MAP-MRI analyses :cite:p:`gu2019`. It is not uncommon to see
parametric maps with patchy or noisy appearances (especially at higher
radial orders); hence smoothing or pooling strategies might be needed,
but these come with trade-offs in resolution. Overall, reproducibility of
MAP-MRI metrics across scanners or sites is still being evaluated, and users
should exercise caution when comparing absolute values across studies.

**Interpretational Caveats**: Unlike biophysical models (e.g., NODDI) that aim
to map specific tissue features, MAP-MRI metrics are phenomenological.
They capture aspects of the diffusion propagator shape but are not uniquely
specific to one biological property. For example, a high NG
(strong non-Gaussianity) could arise from restrictive barriers
(like many small axon fibers) or from mixture effects (such as free water plus
restricted water) – it is a general measure of diffusion complexity.
Similarly, propagator anisotropy (PA) is influenced by multiple factors
(myelination, fiber coherence, etc.). Therefore, one must be careful in
assigning biological interpretations: MAP-MRI metrics should often be
correlated with other modalities or histology to draw firm conclusions.
:cite:t:`jelescu2017design` underscore that such indices "remain an indirect
characterization of microstructure" and lack one-to-one specificity.

**Partial Volume and Free Water Effects**: MAP-MRI metrics can be confounded
by partial voluming with cerebrospinal fluid or lesions. For instance,
CSF-rich voxels will show very low non-Gaussianity (since free water diffusion
is Gaussian) and high apparent diffusivity, which could mask or dilute the
tissue's NG and RTOP values (see also
`DIPY documentation <https://docs.dipy.org/stable/examples_built/reconstruction/reconst_mapmri.html>`_).


**Comparison with Other Models**: Another caveat is how MAP-MRI metrics relate
to more familiar diffusion metrics. There is often moderate correlation between
MAP-MRI indices and :ref:`DTI <dti_model>`/:ref:`DKI <dki_model>`
indices (e.g., PA correlates with FA, NG correlates
with mean kurtosis), meaning they are not independent. However, MAP-MRI can potentially
capture extremes where :ref:`DTI <dti_model>` fails (e.g., very complex fiber regions or very
restrictive environments). Still, when a simpler model explains the data well
(like in a single fiber population), MAP-MRI might not offer significantly
different insights. It's best used when diffusion
data are sufficiently rich (multi-shell, high b) and the tissue complexity
merits a detailed characterization. In situations with limited data or very
homogeneous fiber architecture, simpler models might be more parsimonious.

**Best Practices**: In summary, general neuroimaging users applying MAP-MRI
should (a) ensure high-quality, multi-shell diffusion acquisitions (and be
cautious with fast imaging techniques that haven't been validated for propagator
analysis), (b) use available regularization and constraints to obtain physically
plausible propagators, (c) consider quantifying the uncertainty of the resulting
metrics (e.g., via bootstrap or repeated scans) especially for clinical or
single-subject interpretations, and (d) avoid over-interpreting the metrics in
isolation. Where possible, integrate MAP-MRI findings with other measures
(DTI, myelin imaging, etc.) for a more reliable understanding of the underlying biology.


**********
References
**********

.. bibliography::
   :style: unsrt
   :filter: docname in docnames
