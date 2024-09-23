.. include:: links.rst


.. _input_data:

#######################
Input Data for QSIRecon
#######################

*Raw BIDS dMRI should not be used and will not work*. Instead,
gather data processed by QSIPrep_ or UKBioBank_. The directory
containing the ``sub-*`` directories (if from QSIPrep) or the
directory with the numerals/underscores per subject if UKBB
will be the first argument to QSIRecon.

***********************
Using Data from QSIPrep
***********************

QSIprep outputs will work as inputs for QSIRecon as-is, aside from some
when some specific options were selected for preprocessing.

If ``--anat-modality none`` was used
====================================

If ``--anat-modality none`` was used, there will be no preprocessed T1w data in
the ``qsirecon`` results. Instead the DWI images have been aligned to AC-PC as
closely as possible (likely imperfectly). In this case, the FreeSurfer
skull-stripped ``brain.mgz`` is rigidly registered to ``dwiref`` of each
preprocessed DWI. The FreeSurfer brain mask is resampled to the grid of the DWI.

If structural connectivity is calculated during the reconstruction workflow
(or any atlases are specified in the ``anatomical:`` section of the
workflow's ``.yaml`` file), the coregistered-to-DWI ``brain.mgz`` image will be
normalized to the MNI152NLin2009cAsym template using ``antsRegistration``.
The reverse transform is used to get parcellations aligned to the DWI.

.. _other_pipeline_input:

******************************************
Using Data Preprocessed by Other Pipelines
******************************************

Many open datasets are provided in minimally preprocessed form. Most of these have a
bespoke processing pipeline and in many cases these pipelines are very similar to
QSIRecon. Instead of preprocessing these from scratch, you can run reconstruction
workflows on the minimally preprocessed data by specifying the pipeline that was
used for preprocessing.

UK BioBank Preprocessed Data
============================

To use the UK BioBank preprocessed dMRI data, specify ``--input-type ukb``.
Note that the transforms to/from MNI space are not able to be used at this time.
This means that a new spatial normalization will have to be estimated in order to
use any of the workflows that require one.

HCP Young Adult Preprocessed Data
=================================

To use minimally preprocessed dMRI data from HCP-YA specify ``--input-type hcpya``.
Note that the transforms to/from MNI space are not able to be used at this time. Please note that if you have the
HCPYA dataset from datalad (https://github.com/datalad-datasets/human-connectome-project-openaccess)
then you should ``datalad get`` relevant subject data before running QSIRecon,
and be mindful about how you mount the directory in Docker/Apptainer.

.. _anat_reqs:

********************************************
Anatomical Data for Reconstruction Workflows
********************************************

Some reconstruction workflows require additional anatomical data to work properly.
This table shows which reconstruction workflows depend on the availibility of
anatomical data:


+-----------------------------------------+-------------------+-------------------+--------------+
| Option                                  |    Req. T1w       |  Req. FreeSurfer  |   Req. SDC   |
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
specify the directory containing freesurfer outputs with the ``--freesurfer-input`` flag. If you
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
       --freesurfer-input "${PWD}/derivatives/freesurfer" \
       -v -v

This will read the FreeSurfer data, align it to the ``qsiprep`` results and use it
for subsequent reconstruction steps.


How FreeSurfer Data is Included
===============================

If a T1w image is available in the input data the ``brain.mgz`` image from
freesurfer is registered to the appropriate T1w image.  The transform from
freesurfer native space into alignment with the ``qsirecon`` outputs is achieved
by converting ``brain.mgz`` into NIfTI format and adjusting the affine matrix
such that the images are aligned in world coordinates. This prevents an extra
interpolation.
