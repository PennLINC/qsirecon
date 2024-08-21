.. include:: links.rst

.. _builtin_workflows:

#################################
Built-In Reconstruction Workflows
#################################

The Built-In recon workflows can be easily selected by specifying their name
after the ``--recon-spec`` flag (e.g. ``--recon-spec amico_noddi``).
Many of these workflows were originally described in :footcite:t:`cieslak2021qsiprep`.
Not all workflows are suitable for all kinds of dMRI data.
Be sure to check :ref:`appropriate_schemes`.

By specifying just a name for ``--recon_spec``, you will be using all the default arguments
for the various steps in that workflow. Workflows can be customized
(see :ref:`building_workflows`).


.. note::
  The MRtrix workflows are identical up to the FOD estimation. In each case the fiber response
  function is estimated using ``dwi2response dhollander`` :footcite:p:`dhollander2019response`
  with a mask based on the T1w.
  The main differences are in

    * the CSD algorithm used in dwi2fod (msmt_csd or ss3t_csd)
    * whether a T1w-based tissue segmentation is used during tractography

  In the ``*_noACT`` versions of the pipelines, no T1w-based segmentation is used during
  tractography. Otherwise, cropping is performed at the GM/WM interface, along with backtracking.

  In all pipelines, tractography is performed using
  tckgen_, which uses the iFOD2 probabilistic tracking method to generate 1e7 streamlines with a
  maximum length of 250mm, minimum length of 30mm, FOD power of 0.33. Weights for each streamline
  were calculated using SIFT2_ :footcite:p:`smith2015sift2` and were included for while estimating the
  structural connectivity matrix.


.. warning::
  We don't recommend using ACT with FAST segmentations. The full benefits of ACT
  require very precise tissue boundaries and FAST just doesn't do this reliably
  enough. We strongly recommend the ``hsvs`` segmentation if you're going to
  use ACT. Note that this requires ``--freesurfer-input``

*********
Workflows
*********

.. _mrtrix_multishell_msmt_ACT-hsvs:


``mrtrix_multishell_msmt_ACT-hsvs``
===================================

This workflow uses the ``msmt_csd`` algorithm :footcite:p:`msmt5tt` to estimate FODs for white matter,
gray matter and cerebrospinal fluid using *multi-shell acquisitions*. The white matter FODs are
used for tractography and the T1w segmentation is used for anatomical constraints :footcite:p:`smith2012anatomically`.
The T1w segmentation uses the hybrid surface volume segmentation (hsvs) :footcite:p:`smith2020hybrid` and
requires ``--freesurfer-input``.

.. _mrtrix_multishell_msmt_ACT-fast:

``mrtrix_multishell_msmt_ACT-fast``
===================================

Identical to :ref:`mrtrix_multishell_msmt_ACT-hsvs` except FSL's FAST is used for
tissue segmentation. This workflow is not recommended.


.. _mrtrix_multishell_msmt_noACT:


``mrtrix_multishell_msmt_noACT``
================================

This workflow uses the ``msmt_csd`` algorithm :footcite:p:`msmt5tt` to estimate FODs for white matter,
gray matter and cerebrospinal fluid using *multi-shell acquisitions*. The white matter FODs are
used for tractography with no T1w-based anatomical constraints.


.. _mrtrix_singleshell_ss3t_ACT-hsvs:


``mrtrix_singleshell_ss3t_ACT-hsvs``
====================================

This workflow uses the ``ss3t_csd_beta1`` algorithm :footcite:p:`dhollander2016novel`
to estimate FODs for white matter,
and cerebrospinal fluid using *single shell (DTI) acquisitions*. The white matter FODs are
used for tractography and the T1w segmentation is used for anatomical constraints :footcite:p:`smith2012anatomically`.
The T1w segmentation uses the hybrid surface volume segmentation (hsvs) :footcite:p:`smith2020hybrid` and
requires ``--freesurfer-input``.

.. _mrtrix_singleshell_ss3t_ACT-fast:

``mrtrix_multishell_msmt_ACT-fast``
===================================

Identical to :ref:`mrtrix_singleshell_ss3t_ACT-hsvs` except FSL's FAST is used for
tissue segmentation. This workflow is not recommended.

.. _mrtrix_singleshell_ss3t_noACT:

``mrtrix_singleshell_ss3t_noACT``
=================================

This workflow uses the ``ss3t_csd_beta1`` algorithm :footcite:p:`dhollander2016novel`
to estimate FODs for white matter,
and cerebrospinal fluid using *single shell (DTI) acquisitions*. The white matter FODs are
used for tractography with no T1w-based anatomical constraints.

.. _pyafq_tractometry:

``pyafq_tractometry``
=====================

This workflow uses the AFQ :footcite:p:`pyafq2` implemented in Python :footcite:p:`pyafq` to recognize
major white matter pathways within the tractography, and then extract tissue properties along
those pathways. See the `pyAFQ documentation <https://yeatmanlab.github.io/pyAFQ/>`_ .

.. _pyafq_input_trk:

``mrtrix_multishell_msmt_pyafq_tractometry``
============================================

Identical to :ref:`pyafq_tractometry` except that tractography generated using IFOD2 from MRTrix3,
instead of using pyAFQ's default DIPY tractography.
This can also be used as an example for how to import tractographies from other
reconstruciton pipelines to pyAFQ.


.. _amico_noddi:

``amico_noddi``
===============

This workflow estimates the NODDI :footcite:p:`noddi` model using the implementation from
AMICO :footcite:p:`amico`. Images with intra-cellular volume fraction (ICVF), isotropic volume
fraction (ISOVF), orientation dispersion (OD) are written to outputs. Additionally, a DSI
Studio fib file is created using the peak directions and ICVF as a stand-in for QA to be
used for tractography.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/amico_noddi.csv
   :widths: 15, 10, 30

.. _dsi_studio_gqi:

``dsi_studio_gqi``
==================

Here the standard GQI plus deterministic tractography pipeline is used :footcite:p:`yeh2013deterministic`.
GQI works on
almost any imaginable sampling scheme because DSI Studio will internally interpolate the q-space
data so  symmetry requirements are met. GQI models the water diffusion ODF, so ODF peaks are much
smaller  than you see with CSD. This results in a rather conservative peak detection, which greatly
benefits from having more diffusion data than a typical DTI.

5 million streamlines are created with a maximum length of 250mm, minimum length of 30mm,
random seeding, a step size of 1mm and an automatically calculated QA threshold.

Additionally, a number of anisotropy scalar images are produced such as QA, GFA and ISO.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/dsi_studio_gqi.csv
   :widths: 15, 10, 30


.. _dsi_studio_autotrack:

``dsi_studio_autotrack``
========================

This workflow implements DSI Studio's q-space diffeomorphic reconstruction (QSDR), the MNI space
(ICBM-152) version of GQI, followed by automatic fiber tracking (autotrack) :footcite:p:`autotrack,yeh2022population`
of 56 white matter pathways. Autotrack uses a population-averaged tractography atlas
(based on HCP-Young Adult data) to identify tracts of interest in individual subject's data.
The autotrack procedure seeds deterministic fiber tracking with randomized parameter saturation
within voxels that correspondto each tract in the tractography atlas and determines whether
generated streamlines belong to the target tract based on the Hausdorff distance between
subject and atlas streamlines.

Reconstructed subject-specific tracts are written out as .tck files that are aligned to the
qsirecon-generated _dwiref.nii.gz and preproc_T1w.nii.gz volumes; .tck files can be visualized
overlaid on these volumes in mrview or MI-brain. Note, .tck files will not appear in alignment
with the dwiref/T1w volumes in DSI Studio due to how the .tck files are read in.

Diffusion metrics (e.g., dti_fa, gfa, iso,rdi, nrdi02) and shape statistics (e.g., mean_length,
span, curl, volume, endpoint_radius) are calculated for subject-specific tracts and written out in
an AutoTrackGQI.csv file.

.. _ss3t_autotrack:

``ss3t_autotrack``
========================

This workflow is identical to :ref:`dsi_studio_autotrack`, except it substitutes
the GQI fit with the ``ss3t_csd_beta1`` algorithm :footcite:p:`dhollander2016novel`
to estimate FODs for white matter.

This is a good workflow for doing tractometry on low-quality single shell data.

.. _tortoise:

``TORTOISE``
============

The TORTOISE :footcite:p:`tortoisev3` software can calculate Tensor and MAPMRI fits,
along with their many associated scalar maps.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/tortoise.csv
   :widths: 15, 10, 30


.. _dipy_mapmri:

``dipy_mapmri``
===============

The MAPMRI method is used to estimate EAPs from which ODFs are calculated analytically. This
method produces scalars like RTOP, RTAP, QIV, MSD, etc.

The ODFs are saved in DSI Studio format and tractography is run identically to that in
:ref:`dsi_studio_gqi`.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/dipy_mapmri.csv
   :widths: 15, 10, 30

.. _dipy_dki:

``dipy_dki``
===============

A DKI model is fit to the dMRI signal and multiple scalar maps are produced.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/dipy_dki.csv
   :widths: 15, 10, 30

.. _dipy_3dshore:

``dipy_3dshore``
================

This uses the BrainSuite 3dSHORE basis in a Dipy reconstruction. Much like :ref:`dipy_mapmri`,
a slew of anisotropy scalars are estimated. Here the :ref:`dsi_studio_gqi` fiber tracking is
again run on the 3dSHORE-estimated ODFs.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/dipy_3dshore.csv
   :widths: 15, 10, 30


.. _reorient_fslstd:

``reorient_fslstd``
===================

Reorients the ``qsirecon`` preprocessed DWI and bval/bvec to the standard FSL orientation.
This can be useful if FSL tools will be applied outside of ``qsirecon``.


.. _csdsi_3dshore:

``csdsi_3dshore``
=================

**[EXPERIMENTAL]** This pipeline is for DSI or compressed-sensing DSI. The first step is a
L2-regularized 3dSHORE reconstruction of the ensemble average propagator in each voxel. These EAPs
are then used for two purposes

 1. To calculate ODFs, which are then sent to DSI Studio for tractography
 2. To estimate signal for a multishell (specifically HCP) sampling scheme, which is run
    through the  pipeline

All outputs, including the imputed HCP sequence are saved in the outputs directory.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/csdsi_3dshore.csv
   :widths: 15, 10, 30


.. _hbcd_scalar_maps:

``hbcd_scalar_maps``
====================

Designed to run on [HBCD](https://hbcdstudy.org/) data, this is also a general-purpose
way to get many multishell-supported fitting methods, including

  * :ref:`dipy_dki`
  * :ref:`TORTOISE` (model-MAPMRI)
  * :ref:`TORTOISE` (model-tensor using only b<4000)
  * :ref:`amico_noddi`
  * :ref:`dsi_studio_gqi`

Bundles are generated using :ref:`dsi_studio_autotrack`.
All the scalars generated by these models are then mapped

  1. Into template space
  2. On to the bundles from :ref:`dsi_studio_autotrack`

In total, the scalars estimated by this workflow are:

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/hbcd_scalar_maps.csv
   :widths: 15, 10, 30


.. _appropriate_schemes:

***************************************************
Which workflows are appropriate for your dMRI data?
***************************************************

Most reconstruction workflows will fit a model to the dMRI data. Listed below are
the model-fitting workflows and which sampling schemes work with them.

+-------------------------------------------+-------------+------------+-----------------+
| Name                                      | MultiShell  | Cartesian  |   SingleShell   |
+===========================================+=============+============+=================+
|:ref:`mrtrix_multishell_msmt_ACT-fast`\*   |     Yes     |    No      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_multishell_msmt_ACT-hsvs`     |     Yes     |    No      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_multishell_msmt_noACT`        |     Yes     |    No      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_singleshell_ss3t_noACT`       |     No      |    No      |      Yes        |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-hsvs`    |     No      |    No      |      Yes        |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-fast`\*  |     No      |    No      |      Yes        |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`pyafq_tractometry`                   |     Yes     |    No      |      Yes        |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`pyafq_input_trk`                     |     Yes     |    No      |      Yes        |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`amico_noddi`                         |     Yes     |    No      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`TORTOISE`                            |     Yes     |    No      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`dsi_studio_gqi`                      |     Yes     |   Yes      |    Yes*         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`dsi_studio_autotrack`                |     Yes     |   Yes      |    Yes          |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`ss3t_autotrack`                      |     No      |   No       |    Yes          |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`dipy_mapmri`                         |     Yes     |   Yes      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`dipy_dki`                            |     Yes     |    No      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`dipy_3dshore`                        |     Yes     |   Yes      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`csdsi_3dshore`                       |     Yes     |   Yes      |      No         |
+-------------------------------------------+-------------+------------+-----------------+
|:ref:`hbcd_scalar_maps`                    |     Yes     |    No      |      No         |
+-------------------------------------------+-------------+------------+-----------------+

\* Not recommended


**********
References
**********

.. footbibliography::
