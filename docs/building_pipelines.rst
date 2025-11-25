.. include:: links.rst

.. _building_pipelines:

###############################
Custom Reconstruction Pipelines
###############################

*QSIRecon* pipelines are defined in ``YAML`` files. The :ref:`builtin_pipelines`
can be found in the *QSIRecon* ``YAML`` format on
`github <https://github.com/PennLINC/qsirecon/tree/main/qsirecon/data/pipelines>`_,
and are a great place to get started with customizing parts of these pipelines.

This format has a few key parts.

***********************
Pipeline-level metadata
***********************

At the root level of the Recon Spec there are

.. code-block:: yaml
  name: dsistudio_pipeline
  anatomical:
  - mrtrix_5tt_hsvs
  workflows: []

The ``"name"`` element defines the name of the pipeline. There will be a directory in your working
directory with this name. The ``"anatomical"`` field lists additional things to compute
based on the T1w or T2w images in the input data. Currently accepted values are

  * ``"mrtrix_5tt_hsvs"``: A MRtrix_ 5tt segmentation based on :footcite:t:`smith2020hybrid`.
    This requires :ref:`include_freesurfer`.


******************
Pipeline workflows
******************

A *workflow* in the *QSIRecon* ``"workflows"`` list represents a unit of processing,
implemented as a Nipype workflow.
For example, if you want to perform CSD there are other steps that should
happen before actually fitting spherical harmonics to the data.
The entry for this in the ``"workflows"`` list could look like:

.. code-block:: yaml

  workflows:
  -   action: csd
      name: msmt_csd
      parameters:
          fod:
              algorithm: msmt_csd
              max_sh:
              - 8
              - 8
              - 8
          mtnormalize: true
          response:
              algorithm: dhollander
      qsirecon_suffix: MRtrix3_act-FAST
      software: MRTrix3

*QSIRecon* figures out which software to use based on the values specified
for ``"software"`` and ``"action"``. The full list of recognized actions
for each software package can be found in
:meth:`qsirecon.workflows.recon.build_workflow.workflow_from_spec`.
All workflows must have a ``name`` element, this serves
as an id for this workflow and is used for :ref:`passing_downstream`.


.. _connecting_workflows:

Connecting Workflows
====================

Mixing between software packages is something *QSIRecon* generally does well.
There are a number of ways that workflows can exchange data with one another.

.. _passing_downstream:

Passing method-specific outputs from one workflow as inputs to another
------------------------------------------------------------------

When one workflow produces outputs that are specifically used as inputs for
another workflow, you can pass them by matching the ``"input"`` field to the
name of the upstream workflow. Here is an example connecting a CSD calculation
to MRtrix3 tractography.

.. code-block:: yaml

  workflows:
  -   action: csd
      name: msmt_csd
      qsirecon_suffix: MRtrix3_act-FAST
      software: MRTrix3

  -   action: tractography
      input: msmt_csd
      name: track_ifod2
      qsirecon_suffix: MRtrix3_act-FAST
      software: MRTrix3

.. note::
    There can only be zero (inputs come from the input data) or one
    (inputs come from a mixture of the input data and the "input" workflow)
    name specified for ``"input"``.

.. _scalars_resampling:

Mapping scalar data to different output spaces
----------------------------------------------

Most pipelines produce interesting maps of model-derived parameters.
These parameters are calculated in subject native space, which is not
particularly useful for statistics. You can map these scalars to standard
spaces with a "template mapper". Suppose I wanted to fit a NODDI model and
a DKI model. Then I wanted to transform their model-derived parameters to
the template space used in QSIPrep. My ``"workflows"`` might look like:

.. code-block:: yaml

  workflows:
  -   action: DKI_reconstruction
      name: dipy_dki
      qsirecon_suffix: DIPYDKI
      software: Dipy
  -   action: fit_noddi
      name: fit_noddi
      qsirecon_suffix: NODDI
      software: AMICO
  -   action: template_map
      input: qsirecon
      name: template_map
      parameters:
          interpolation: NearestNeighbor
      scalars_from:
      - dipy_dki
      - gqi_scalars
      software: qsirecon

By listing the names of the scalar-producing workflows in the ``"scalars_from"`` field
you will end up with the scalars in both subject native and template space in the
output directory for each workflow that produces the scalars.

.. _scalars_to_bundles:

Mapping scalar data to bundles
------------------------------

Perhaps the most biologically meaningful unit of analysis for dMRI is
the bundle. Much like :ref:`scalars_resampling`, the scalar maps
produced by workflows can be summarized along bundles. The requirements
for bundle mapping are

 1. A bundle-creating workflow (such as autotrack or pyafq)
 2. A scalar-producing workflow
 3. A bundle_means workflow that sets up the mapping

Here is a small example where we use autotrack bundles:

.. code-block:: yaml

  workflows:
  -   action: DKI_reconstruction
      name: dipy_dki
      qsirecon_suffix: DIPYDKI
      software: Dipy
  -   action: fit_noddi
      name: fit_noddi
      qsirecon_suffix: NODDI
      software: AMICO
  -   action: autotrack
      input: dsistudio_gqi
      name: autotrackgqi
      qsirecon_suffix: DSIStudio
      software: DSI Studio
  -   action: bundle_map
      input: autotrackgqi
      name: bundle_means
      scalars_from:
      - dipy_dki
      - fit_noddi
      software: qsirecon

This will produce a tsv with the NODDI and DKI scalars summarized for
each bundle produced by autotrack.


**********
References
**********

.. footbibliography::
