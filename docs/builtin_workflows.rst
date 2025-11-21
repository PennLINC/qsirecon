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


*********
Workflows
*********

MRtrix3-based Workflows
=======================

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
  use ACT. Note that this requires ``--fs-subjects-dir``


.. _mrtrix_dwi_outputs:

MRtrix3 DWI Outputs
-------------------
These files are located in the ``dwi/`` directories.

.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/mrtrix_dwi.csv
   :widths: 15, 30


.. _mrtrix_anatomical_outputs:

MRtrix3 Anatomical Outputs
--------------------------
These files are located ``anat/`` directories.

.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/mrtrix_anat.csv
   :widths: 15, 30


.. _mrtrix_multishell_msmt_ACT-hsvs:

``mrtrix_multishell_msmt_ACT-hsvs``
===================================

This workflow uses the ``msmt_csd`` algorithm :footcite:p:`msmt5tt` to estimate FODs for white matter,
gray matter and cerebrospinal fluid using *multi-shell acquisitions*. The white matter FODs are
used for tractography and the T1w segmentation is used for anatomical constraints :footcite:p:`smith2012anatomically`.
The T1w segmentation uses the hybrid surface volume segmentation (hsvs) :footcite:p:`smith2020hybrid` and
requires ``--fs-subjects-dir``.
This workflow produces :ref:`mrtrix_dwi_outputs` and :ref:`mrtrix_anatomical_outputs`.

.. _mrtrix_multishell_msmt_ACT-fast:

``mrtrix_multishell_msmt_ACT-fast``
===================================

Identical to :ref:`mrtrix_multishell_msmt_ACT-hsvs` except FSL's FAST is used for
tissue segmentation. This workflow is not recommended.
This workflow produces :ref:`mrtrix_dwi_outputs`.


.. _mrtrix_multishell_msmt_noACT:


``mrtrix_multishell_msmt_noACT``
================================

This workflow uses the ``msmt_csd`` algorithm :footcite:p:`msmt5tt` to estimate FODs for white matter,
gray matter and cerebrospinal fluid using *multi-shell acquisitions*. The white matter FODs are
used for tractography with no T1w-based anatomical constraints.
This workflow produces :ref:`mrtrix_dwi_outputs`.


.. _mrtrix_singleshell_ss3t_ACT-hsvs:


``mrtrix_singleshell_ss3t_ACT-hsvs``
====================================

This workflow uses the ``ss3t_csd_beta1`` algorithm :footcite:p:`dhollander2016novel`
to estimate FODs for white matter,
and cerebrospinal fluid using *single shell (DTI) acquisitions*. The white matter FODs are
used for tractography and the T1w segmentation is used for anatomical constraints :footcite:p:`smith2012anatomically`.
The T1w segmentation uses the hybrid surface volume segmentation (hsvs) :footcite:p:`smith2020hybrid` and
requires ``--fs-subjects-dir``.
This workflow produces :ref:`mrtrix_dwi_outputs` and :ref:`mrtrix_anatomical_outputs`.

.. _mrtrix_singleshell_ss3t_ACT-fast:

``mrtrix_singleshell_ss3t_ACT-fast``
====================================

Identical to :ref:`mrtrix_singleshell_ss3t_ACT-hsvs` except FSL's FAST is used for
tissue segmentation. This workflow is not recommended.
This workflow produces :ref:`mrtrix_dwi_outputs`.

.. _mrtrix_singleshell_ss3t_noACT:

``mrtrix_singleshell_ss3t_noACT``
=================================

This workflow uses the ``ss3t_csd_beta1`` algorithm :footcite:p:`dhollander2016novel`
to estimate FODs for white matter,
and cerebrospinal fluid using *single shell (DTI) acquisitions*. The white matter FODs are
used for tractography with no T1w-based anatomical constraints.
This workflow produces :ref:`mrtrix_dwi_outputs`.

.. _pyafq_tractometry:

``pyafq_tractometry``
=====================

This workflow uses the AFQ :footcite:p:`pyafq2` implemented in Python :footcite:p:`pyafq` to recognize
major white matter pathways within the tractography, and then extract tissue properties along
those pathways. See the `pyAFQ documentation <https://yeatmanlab.github.io/pyAFQ/>`_ .

PyAFQ Outputs
-------------

+------------------------+-------------------------------------------+
| File Name              | Description                               |
+========================+===========================================+
| sub-* (directory)      | PyAFQ results direcrory for each subject  |
+------------------------+-------------------------------------------+


.. _mrtrix_multishell_msmt_pyafq_tractometry:

``mrtrix_multishell_msmt_pyafq_tractometry``
============================================

Identical to :ref:`pyafq_tractometry` except that tractography generated using IFOD2 from MRTrix3,
instead of using pyAFQ's default DIPY tractography.
This can also be used as an example for how to import tractographies from other
reconstruciton pipelines to pyAFQ.
This workflow produces :ref:`mrtrix_dwi_outputs`.

PyAFQ Outputs
-------------

+------------------------+-------------------------------------------+
| File Name              | Description                               |
+========================+===========================================+
| sub-* (directory)      | PyAFQ results direcrory for each subject  |
+------------------------+-------------------------------------------+


.. _amico_noddi:

``amico_noddi``
===============

This workflow estimates the NODDI :footcite:p:`noddi` model using the implementation from
AMICO :footcite:p:`amico` and tissue fraction modulation described in :footcite:p:`parker2021not`.
Images with (modulated) intra-cellular volume fraction (ICVF), isotropic volume fraction (ISOVF),
(modulated) orientation dispersion (OD), root mean square error (RMSE) and normalized RMSE are written to outputs.
Additionally, a DSI Studio fib file is created using the peak directions and ICVF as a stand-in for QA to be
used for tractography.

Please see Parker 2021 :footcite:p:`parker2021not` for a detailed description of use and application of the
tissue fraction modulated outputs.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/amico_noddi.csv
   :widths: 15, 10, 30

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/amico_noddi.csv
   :widths: 15, 30


.. _dsi_studio_gqi:

``dsi_studio_gqi``
==================

Here the standard GQI plus deterministic tractography pipeline is used :footcite:p:`yeh2013`.
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

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/dsistudio_gqi.csv
   :widths: 15, 30


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
*QSIRecon*-generated _dwiref.nii.gz and preproc_T1w.nii.gz volumes; .tck files can be visualized
overlaid on these volumes in mrview or MI-brain. Note, .tck files will not appear in alignment
with the dwiref/T1w volumes in DSI Studio due to how the .tck files are read in.

Diffusion metrics (e.g., dti_fa, gfa, iso,rdi, nrdi02) and shape statistics (e.g., mean_length,
span, curl, volume, endpoint_radius) are calculated for subject-specific tracts and written out in
an AutoTrackGQI.csv file.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/dsi_studio_gqi.csv
   :widths: 15, 10, 30

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/dsistudio_autotrack.csv
   :widths: 15, 30


.. _ss3t_fod_autotrack:

``ss3t_fod_autotrack``
======================

This workflow is identical to :ref:`dsi_studio_autotrack`, except it substitutes
the GQI fit with the ``ss3t_csd_beta1`` algorithm :footcite:p:`dhollander2016novel`
to estimate FODs for white matter.

A GQI reconstruction is performed first based on the entire input data.
The QA and ISO images from GQI are used to register the ACPC data to DSI Studio's ICBM 152 template.
The GQI-based registration is used to transform the template bundles to subject ACPC space,
where the SS3T-based FODs are used for tractography.

This is a good workflow for doing tractometry on low-quality single shell data.
If more than one shell is present in the input data, only the highest b-value shell is used.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/dsi_studio_gqi.csv
   :widths: 15, 10, 30

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/ss3t_fod_autotrack.csv
   :widths: 15, 30


.. _tortoise:

``TORTOISE``
============

The TORTOISE :footcite:p:`tortoisev3` software can calculate Tensor and MAPMRI fits,
along with their many associated scalar maps. This workflow only produces scalar maps.

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/tortoise.csv
   :widths: 15, 10, 30

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/tortoise.csv
   :widths: 15, 30


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

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/dipy_mapmri.csv
   :widths: 15, 30


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

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/dipy_dki.csv
   :widths: 15, 30


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

Reorients the *QSIRecon* preprocessed DWI and bval/bvec to the standard FSL orientation.
This can be useful if FSL tools will be applied outside of *QSIRecon*.


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

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/csdsi_3dshore.csv
   :widths: 15, 30


.. _hbcd_scalar_maps:

``hbcd_scalar_maps``
====================

Designed to run on `HBCD <https://hbcdstudy.org/>`_ data, this is also a general-purpose
way to get many multishell-supported fitting methods, including

  * :ref:`dipy_dki`
  * :ref:`TORTOISE` (model-MAPMRI)
  * :ref:`TORTOISE` (model-tensor using only b<4000)
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

Other Outputs
-------------
.. csv-table::
   :header: "File Name", "Description"
   :file: nonscalars/hbcd_scalar_maps.csv
   :widths: 15, 30


.. _multishell_scalarfest:

``multishell_scalarfest``
=========================

This is a general-purpose way to get scalar maps from many multishell-supported fitting methods,
including:

  * :ref:`dipy_dki`
  * :ref:`TORTOISE` (model-MAPMRI)
  * :ref:`TORTOISE` (model-tensor using only b<4000)
  * :ref:`dsi_studio_gqi`
  * :ref:`amico_noddi`

Scalar Maps
-----------
.. csv-table::
   :header: "Model", "Parameter", "Description"
   :file: recon_scalars/multishell_scalarfest.csv
   :widths: 15, 10, 30

Other Outputs
-------------

No other outputs are produced.

.. _appropriate_schemes:

***************************************************
Which workflows are appropriate for your dMRI data?
***************************************************

Most reconstruction workflows will fit a model to the dMRI data. Listed below are
the model-fitting workflows and which sampling schemes work with them.

+------------------------------------------------+-------------+------------+-----------------+
| Name                                           | MultiShell  | Cartesian  |   SingleShell   |
+================================================+=============+============+=================+
|:ref:`amico_noddi`                              |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`csdsi_3dshore`                            |     Yes     |    Yes     |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`dipy_3dshore`                             |     Yes     |    Yes     |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`dipy_dki`                                 |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`dipy_mapmri`                              |     Yes     |    Yes     |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`dsi_studio_autotrack`                     |     Yes     |    Yes     |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`dsi_studio_gqi`                           |     Yes     |    Yes     |      Yes*       |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`hbcd_scalar_maps`                         |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_multishell_msmt_ACT-fast`\*        |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_multishell_msmt_ACT-hsvs`          |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_multishell_msmt_noACT`             |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_multishell_msmt_pyafq_tractometry` |     Yes     |    No      |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-fast`\*       |     No      |    No      |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-hsvs`         |     No      |    No      |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`mrtrix_singleshell_ss3t_noACT`            |     No      |    No      |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`multishell_scalarfest`                    |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`pyafq_tractometry`                        |     Yes     |    No      |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`reorient_fslstd`                          |     Yes     |    Yes     |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`ss3t_fod_autotrack`                       |     Yes     |    No      |      Yes        |
+------------------------------------------------+-------------+------------+-----------------+
|:ref:`TORTOISE`                                 |     Yes     |    No      |      No         |
+------------------------------------------------+-------------+------------+-----------------+

\* Not recommended

.. _connectivity_matrices:

*********************
Connectivity matrices
*********************

Instead of offering a bewildering number of options for constructing connectivity matrices,
*QSIRecon* will construct as many connectivity matrices as it can given the reconstruction
methods.
It is **highly** recommended that you pick a weighting scheme before you run
these pipelines and only look at those numbers.
If you look at more than one weighting method be sure to adjust your statistics for the
additional comparisons.

To skip this step in your workflow, you can modify an existing recon pipeline by removing the
``action: connectivity`` section from the yaml file.

.. _connectivity_atlases:

Atlases
=======

The following atlases are included in *QSIRecon*.
This means you do not need to add a ``--datasets`` argument to your command line,
and can instead select them just with ``--atlases``.

If you previously were using the default atlases in a "connectivity matrix" workflow,
you can match the previous behavior by adding

``--atlases 4S156Parcels 4S256Parcels 4S456Parcels Brainnetome246Ext AICHA384Ext Gordon333Ext AAL116``

If you use one of them please be sure to cite the relevant publication.

 * ``Brainnetome246Ext``: :footcite:t:`fan2016human`, extended with subcortical parcels.
 * ``AICHA384Ext``: :footcite:t:`joliot2015aicha`, extended with subcortical parcels.
 * ``Gordon333Ext``: :footcite:t:`gordon2016generation`, extended with subcortical parcels.
 * ``AAL116``: :footcite:t:`tzourio2002automated`

The *QSIRecon* atlas set can be downloaded directly from
`box <https://upenn.box.com/shared/static/5k1tvg6soelxdhi9nvrkry6w0z49ctne.xz>`_.

The 4S atlas combines the Schaefer 2018 cortical atlas (version v0143) :footcite:p:`Schaefer_2017`
at 10 different resolutions (100, 200, 300, 400, 500, 600, 700, 800, 900, and 1000 parcels) with
the CIT168 subcortical atlas :footcite:p:`pauli2018high`,
the Diedrichson cerebellar atlas :footcite:p:`king2019functional`,
the HCP thalamic atlas :footcite:p:`najdenovska2018vivo`,
and the amygdala and hippocampus parcels from the HCP CIFTI subcortical parcellation
:footcite:p:`glasser2013minimal`.
The 4S atlas is used in the same manner across three PennLINC BIDS Apps:
*QSIRecon*, *XCP-D*, and *ASLPrep*, to produce synchronized outputs across modalities.
For more information about the 4S atlas, please see https://github.com/PennLINC/AtlasPack.

Atlases are written out to the ``atlases`` subfolder, following
`BEP038 <https://docs.google.com/document/d/1RxW4cARr3-EiBEcXjLpSIVidvnUSHE7yJCUY91i5TfM/edit?usp=sharing>`_.

.. code-block::

   qsirecon/
      atlases/
         dataset_description.json
         atlas-<label>/
            atlas-<label>_space-<label>_res-<label>_dseg.nii.gz
            atlas-<label>_space-<label>_res-<label>_dseg.json
            atlas-<label>_dseg.tsv

Additionally, each atlas is warped to the subject's anatomical space and written out in the
associated reconstruction workflows dataset.

.. code-block::

   qsirecon/
      derivatives/
         qsirecon-<suffix>/
            sub-<label>/
               dwi/
                  sub-<label>_space-ACPC_seg-<label>_dseg.nii.gz
                  sub-<label>_space-ACPC_seg-<label>_dseg.mif.gz
                  sub-<label>_space-ACPC_seg-<label>_dseg.json
                  sub-<label>_space-ACPC_seg-<label>_dseg.tsv


.. _custom_atlases:

Using custom atlases
--------------------

It's possible to use your own atlases provided you organize the atlases into BIDS-Atlas datasets.
Users can control which atlases are used with the ``--atlases`` and ``--datasets`` parameters.

The nifti images should be registered to the
`MNI152NLin2009cAsym <https://github.com/PennLINC/qsirecon/blob/main/qsirecon/data/mni_1mm_t1w_lps.nii.gz>`_
included in *QSIRecon*.
It is essential that your images are in the LPS+ orientation and have the sform zeroed-out in the header.
**Be sure to check for alignment and orientation** in your outputs.


Connectivity Measures
=====================

Connectivity measures are bundled together in binary ``.mat`` files,
rather than as atlas- and measure-specific tabular files.

.. warning::

   We ultimately plan to organize the connectivity matrices according to the BIDS-Connectivity BEP,
   wherein each measure from each atlas is stored in a separate file.

   Therefore, this organization will change in the future.

.. code-block::

   qsirecon/
      derivatives/
         qsirecon-<suffix>/
            sub-<label>/[ses-<label>/]
               dwi/
                  <source_entities>_connectivity.mat

The ``.mat`` file contains a dictionary with all of the connectivity measures specified
by the recon spec for all of the different atlases specified by the user.

For example, in the case where a user has selected a single atlas (``<atlas>``) and
the recon spec specifies a single connectivity measure (``<measure>``),
the ``.mat`` file will contain the following keys:

.. code-block::

   command                                # The command that was run
   atlas_<atlas>_region_ids               # The region ids for the atlas (1 x n_parcels array)
   atlas_<atlas>_region_labels            # The region labels for the atlas (1 x n_parcels array)
   atlas_<atlas>_<measure>_connectivity   # The connectivity matrix for the atlas and measure (n_parcels x n_parcels array)


MRtrix3 Connectivity Measures
-----------------------------

MRtrix3 connectivity workflows produce 4 structural connectome outputs for each atlas.
The 4 connectivity matrix outputs are

   * *atlas_<atlas>_radius<N>.count.connectivity*: raw streamline count based matrix
   * *atlas_<atlas>_sift.radius<N>.count.connectivity*: sift-weighted streamline count based matrix
   * *atlas_<atlas>_radius<N>.meanlength.connectivity*: a matrix containing mean length of raw streamlines
   * *atlas_<atlas>_sift.radius<N>.meanlength.connectivity*: a matrix containing mean length of sifted output

The number ``N`` in ``radiusN`` indicates how many mm the algorithm would search up from a
given streamline's endpoint for a cortical region. E.g., a radius of 2 indicates that
if a streamline ended before hitting gray matter, the search for a cortical
termination region could be up to 2mm from the endpoint.


DSI Studio Connectivity Measures
--------------------------------

DSI Studio has two options for how to count streamlines as "connnecting" a region pair.
``pass`` counts a connection if any part of a streamline intersects two regions.
``end`` requires that a streamline terminates in each region in order to be connected.
There are some practical considerations with each choice:
``pass`` could produce a connectivity matrix with *more* counts than the number of streamlines you requested.
``end`` will include many fewer counts than the streamlines you requested.
Due to the arbitrary nature of streamline tractography, the ``pass`` method is probably more realistic.

Once the streamlines connecting each region pair are found, they need to be used to quantify that connection somehow.
The streamlines connecting a region pair can by default are summarized by

  * *count*: the count of streamlines
  * *ncount*: the count of streamlines normalized by their length
  * *mean_length*: the mean length of streamlines in millimeters
  * *gfa*: the mean Generalized Fractional Anisotropy along the streamlines

A great walkthrough of connectivity analysis with DSI Studio can be found
`here <https://dsi-studio.labsolver.org/doc/gui_t3_whole_brain.html>`_.


**********
References
**********

.. footbibliography::
