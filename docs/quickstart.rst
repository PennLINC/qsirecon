.. include:: links.rst

Quick Start
-----------

There are many options for running ``qsirecon``, and not all of them are suited for
all kinds of processed dMRI data. Suppose the following data is available
in the BIDS input::

  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-01_dwi.nii.gz
  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-01_dwi.nii.gz
  sub-1/ses-1/dwi/sub-1_ses-1_acq-multishell_run-03_dwi.nii.gz
  sub-1/ses-1/fmap/sub-1_ses-1_dir-PA_epi.nii.gz

One way to process these data would be to call ``qsirecon`` like this::

  qsirecon \
    /path/to/inputs /path/to/outputs participant \
    --output-resolution 1.2 \
    --fs-license-file /path/to/license.txt


Specifying outputs
==================

.. note::
   This section covers ``--output-resolution 1.2``, and
   ``--skip-t1-based-spatial-normalization``.

Unlike with fMRI, which can be coregistered to a T1w image and warped to a
template using the T1w image's spatial normalization, the T1w images do not
contain enough contrast to accurately align white matter structures to a
template. For this reason, spatial normalization is typically done *after*
models are fit. Therefore we omit the ``--output-spaces`` argument from
preprocessing: i.e. **no template warping takes place by default**.


The ``--output-resolution`` argument determines the spatial resolution of the
preprocessed dwi series. You can specify the resolution of the original data
or choose to upsample the dwi to a higher spatial resolution. Some
post-processing pipelines such as fixel-based analysis recommend resampling
your output to at least 1.3mm resolution. By choosing this resolution here,
it means your data will only be interpolated once: head motion correction,
susceptibility distortion correction, coregistration and upsampling will be
done in a single step. If your are upsampling your data by more than 10%,
QSIRecon will use BSpline interpolation instead of Lanczos windowed sinc
interpolation.
