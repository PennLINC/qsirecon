.. include:: links.rst


.. _input_data:

#########################
Input Data for *QSIRecon*
#########################

*Raw BIDS dMRI should not be used and will not work*. Instead,
gather data processed by QSIPrep_ or UKBioBank_. The directory
containing the ``sub-*`` directories (if from QSIPrep) or the
directory with the numerals/underscores per subject if UKBB
will be the first argument to *QSIRecon*.

*************************
Using Data from *QSIPrep*
*************************

*QSIPrep* outputs will work as inputs for *QSIRecon* as-is, aside from some
when some specific options were selected for preprocessing.

.. important::

    *QSIRecon* endeavors to maintain compatibility with *QSIPrep* outputs from the same
    release series.
    We cannot guarantee that *QSIPrep* outputs from different releases will work
    with *QSIRecon*.

    *QSIPrep* and *QSIRecon* follow a matched release schedule,
    with major (i.e., backwards-incompatible) releases made at the same time.
    Minor and patch releases may be made independently.


.. _anatomical_image:

Which anatomical image is used
==============================

*QSIRecon* uses one preprocessed anatomical image per subject
(or per session, when *QSIPrep* was run with session-wise anatomical processing)
as the spatial reference for anatomical data.
It looks in the ``anat`` folder for a ``desc-preproc`` ``T1w`` or ``T2w`` image in
``ACPC`` space, along with the brain mask (``desc-brain_mask``),
the tissue segmentations (``dseg``), and the transforms between ``ACPC`` and
template space (``from-ACPC_to-<template>`` and ``from-<template>_to-ACPC``).
The mask, segmentations, and transforms are not modality-specific,
so they are used the same way regardless of which image was found.

- **T1w only** (*QSIPrep* run with ``--anat-modality T1w``): the T1w is used.
- **T2w only** (*QSIPrep* run with ``--anat-modality T2w``): the T2w is used.
  Nothing else changes.
- **T1w and T2w**: when *QSIPrep* is run with ``--anat-modality T1w`` on a dataset that
  also contains a T2w, it writes the T2w coregistered to the T1w
  (``space-ACPC_desc-preproc_T2w``) next to the T1w.
  *QSIRecon* uses the T1w, because the mask, segmentations, and transforms were derived
  from it.
  To use the T2w instead, pass a ``--bids-filter-file`` containing::

      {"acpc_preproc": {"suffix": "T2w"}}

  The keys of the filter file are the query names in ``qsirecon/data/io_spec.yaml``.
- **No anatomical image** (*QSIPrep* run with ``--anat-modality none``):
  see the next section.

The anatomical image is used as the registration target for FreeSurfer data
(see :ref:`include_freesurfer`), as the underlay for the scalar-map figures in the
reports, and to name some anatomical outputs.
Reconstruction workflows that require anatomical data (see
:ref:`anatomical_requirements`) accept either modality.


If ``--anat-modality none`` was used
====================================

If ``--anat-modality none`` was used, there will be no preprocessed anatomical data in
the *QSIRecon* results. Instead the DWI images have been aligned to AC-PC as
closely as possible (likely imperfectly).

If FreeSurfer data are provided (see :ref:`include_freesurfer`), the FreeSurfer
skull-stripped ``brain.mgz`` is rigidly registered to the ``dwiref`` image of each
preprocessed DWI, and the FreeSurfer ``aseg`` is used as the brain mask on the grid
of that DWI. A hybrid surface/volume segmentation (``mrtrix_5tt_hsvs``) requested by the
reconstruction workflow is aligned to each DWI in the same way and written as a
derivative of that DWI.

If no FreeSurfer data are provided, a brain mask is estimated from the b=0 images of
each DWI instead.

Without the anatomical-to-template transforms that *QSIPrep* derives from an
anatomical image, nothing can be mapped between template space and the DWI.
Atlases specified with ``--atlases``, structural connectivity, and ``template_map``
nodes therefore cannot be used with ``--anat-modality none`` data, and *QSIRecon*
stops with an error that names the missing transform.

.. _other_pipeline_input:

******************************************
Using Data Preprocessed by Other Pipelines
******************************************

Many open datasets are provided in minimally preprocessed form. Most of these have a
bespoke processing pipeline and in many cases these pipelines are very similar to
*QSIRecon*. Instead of preprocessing these from scratch, you can run reconstruction
workflows on the minimally preprocessed data by specifying the pipeline that was
used for preprocessing.

UK BioBank Preprocessed Data
============================

To use the UK BioBank preprocessed dMRI data, specify ``--input-type ukb``.
Note that the transforms to/from MNI space are not able to be used at this time.
This means that a new spatial normalization will have to be estimated in order to
use any of the workflows that require one.
Also note that only DSI_Studio recon-specs have been tested for UKB inputs.

HCP Young Adult Preprocessed Data
=================================

To use minimally preprocessed dMRI data from HCP-YA specify ``--input-type hcpya``.
Note that the transforms to/from MNI space are not able to be used at this time. Please note that if you have the
HCPYA dataset from datalad (https://github.com/datalad-datasets/human-connectome-project-openaccess)
then you should ``datalad get`` relevant subject data before running *QSIRecon*,
and be mindful about how you mount the directory in Docker/Apptainer.

.. _anat_reqs:

********************************************
.. _anatomical_requirements:

Anatomical Data for Reconstruction Workflows
********************************************

Some reconstruction workflows require additional anatomical data to work properly.
This table shows which reconstruction workflows depend on the availability of
anatomical data.
"Req. anatomical" means a preprocessed T1w or T2w image and brain mask
(see :ref:`anatomical_image`).


+-----------------------------------------+-------------------+-------------------+--------------+
| Option                                  |  Req. anatomical  |  Req. FreeSurfer  |   Req. SDC   |
+=========================================+===================+===================+==============+
|:ref:`mrtrix_multishell_msmt_ACT-hsvs`   |       Yes         |       Yes         |    Yes       |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_multishell_msmt_ACT-fast`   |       Yes         |       No          |    Yes       |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_multishell_msmt_noACT`      |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-hsvs`  |       Yes         |       Yes         |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-fast`  |       Yes         |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_singleshell_ss3t_noACT`     |       Yes         |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`pyafq_tractometry`                 |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`pyafq_input_trk`                   |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`amico_noddi`                       |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`dsi_studio_gqi`                    |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`dipy_mapmri`                       |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`dipy_3dshore`                      |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`csdsi_3dshore`                     |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`reorient_fslstd`                   |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+


.. _include_freesurfer:

Including FreeSurfer Data
=========================

Suppose you ran FreeSurfer on your data (e.g. as part of fmriprep). You can
specify the directory containing freesurfer outputs with the ``--fs-subjects-dir`` flag. If you
have::

    derivatives/freesurfer/sub-x
    derivatives/freesurfer/sub-y
    derivatives/freesurfer/sub-z

and (for example) processed data from QSIPrep::

    derivatives/qsiprep/sub-x
    derivatives/qsiprep/sub-y
    derivatives/qsiprep/sub-z

You can run:

.. code-block:: bash

   apptainer run \
       --containall \
       --writable-tmpfs \
       -B "${PWD}" \
       -B "${FREESURFER_HOME}/license.txt":/license.txt \
       "${PWD}/derivatives/qsiprep" \
       "${PWD}/derivatives/qsirecon" \
       participant \
       -w "${PWD}/work" \
       --nthreads 8 \
       --omp-nthreads 8 \
       --fs-license-file /license.txt \
       --recon-spec mrtrix_multishell_msmt_ACT-hsvs \
       --fs-subjects-dir "${PWD}/derivatives/freesurfer" \
       -v -v

This will read the FreeSurfer data, align it to the *QSIPrep* results and use it
for subsequent reconstruction steps.


How FreeSurfer Data is Included
===============================

If a preprocessed anatomical image (T1w or T2w, see :ref:`anatomical_image`) is
available in the input data, the ``brain.mgz`` image from FreeSurfer is rigidly
registered to it with a mutual-information metric, so either contrast works as the
target. If no anatomical image is available, ``brain.mgz`` is registered to the
``dwiref`` image of each DWI instead. The transform from
freesurfer native space into alignment with the ``qsirecon`` outputs is achieved
by converting ``brain.mgz`` into NIfTI format and adjusting the affine matrix
such that the images are aligned in world coordinates. This prevents an extra
interpolation.
