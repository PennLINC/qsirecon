.. include:: links.rst

.. _building_workflows:

###############################
Custom Reconstruction Workflows
###############################

*QSIRecon* workflows are defined in ``YAML`` files. The :ref:`builtin_workflows`
can be found in the *QSIRecon* ``YAML`` format on
`github <https://github.com/PennLINC/qsirecon/tree/main/qsirecon/data/pipelines>`_,
and are a great place to get started with customizing parts of these workflows.

This format has a few key parts.

***********************
Pipeline-level metadata
***********************

At the root level of the Recon Spec there are

.. code-block:: yaml

  name: dsistudio_pipeline
  anatomical:
  - mrtrix_5tt_hsvs
  nodes: []

The ``"name"`` element defines the name of the pipeline. There will be a directory in your working
directory with this name. The ``"anatomical"`` field lists additional things to compute
based on the T1w or T2w images in the input data. Currently accepted values are

  * ``"mrtrix_5tt_hsvs"``: A MRtrix_ 5tt segmentation based on :footcite:t:`smith2020hybrid`.
    This requires :ref:`include_freesurfer`.


**************
Pipeline nodes
**************

A *node* in the *QSIRecon* ``"nodes"`` list represents a unit of processing.
Unlike NiPype nodes, which represent a single call to a commandline program
or function, a node here represents a set of steps that accomplish a task.
For example, if you want to perform CSD there are other steps that should
happen before actually fitting spherical harmonics to the data.
The entry for this in the ``"nodes"`` list could look like:

.. code-block:: yaml

  nodes:
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
:meth:`qsirecon.workflows.build_workflow.workflow_from_spec`.
All nodes must have a ``name`` element, this serves
as an id for this node and is used for :ref:`passing_downstream`.


.. _precomputed_response_functions:

Using precomputed response functions
====================================

Instead of estimating fiber response functions for each scan,
an MRtrix3 ``csd`` node can load precomputed (e.g., group-average) response functions.
Set the response ``algorithm`` to ``precomputed`` and give the path to each
MRtrix-format response function file:

.. code-block:: yaml

  nodes:
  -   action: csd
      name: msmt_csd
      parameters:
          fod:
              algorithm: msmt_csd
          mtnormalize: true
          response:
              algorithm: precomputed
              wm_txt: /responses/wm.txt
              gm_txt: /responses/gm.txt
              csf_txt: /responses/csf.txt
      qsirecon_suffix: MRtrix3_act-None_response-common
      software: MRTrix3

These paths must be absolute, and they are used exactly as written.
When running *QSIRecon* in a container, the paths refer to locations *inside* the container,
so mount the directory holding the response functions at the matching location
(e.g., ``-v /path/to/my/responses:/responses:ro`` for Docker,
or ``-B /path/to/my/responses:/responses`` for Apptainer/Singularity).

The built-in ``mrtrix_multishell_msmt_noACT_apply`` workflow reads its response functions from
``/responses/wm.txt``, ``/responses/gm.txt``, and ``/responses/csf.txt``.
To use files at other locations, copy that recon spec and edit the paths.

.. note::
    Earlier versions of *QSIRecon* located these files with the ``--recon-spec-aux-files``
    command-line option.
    That option has been removed; write the full paths in the recon spec instead.


.. _connecting_nodes:

Connecting Nodes
================

Mixing between software packages is something *QSIRecon* generally does well.
There are a number of ways that nodes can exchange data with one another.

.. _passing_downstream:

Passing method-specific outputs from one node as inputs to another
------------------------------------------------------------------

When one node produces outputs that are specifically used as inputs for
another node, you can pass them by matching the ``"input"`` field to the
name of the upstream node. Here is an example connecting a CSD calculation
to MRtrix3 tractography.

.. code-block:: yaml

  nodes:
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
    (inputs come from a mixture of the input data and the "input" node)
    name specified for ``"input"``.

.. _scalars_resampling:

Mapping scalar data to different output spaces
----------------------------------------------

Most workflows produce interesting maps of model-derived parameters.
These parameters are calculated in subject native space, which is not
particularly useful for statistics. You can map these scalars to standard
spaces with a "template mapper". Suppose I wanted to fit a NODDI model and
a DKI model. Then I wanted to transform their model-derived parameters to
the template space used in QSIPrep. My ``"nodes"`` might look like:

.. code-block:: yaml

  nodes:
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

By listing the names of the scalar-producing nodes in the ``"scalars_from"`` field
you will end up with the scalars in both subject native and template space in the
output directory for each node that produces the scalars.

.. _scalars_to_bundles:

Mapping scalar data to bundles
------------------------------

Perhaps the most biologically meaningful unit of analysis for dMRI is
the bundle. Much like :ref:`scalars_resampling`, the scalar maps
produced by nodes can be summarized along bundles. The requirements
for bundle mapping are

 1. A bundle-creating node (such as autotrack or pyafq)
 2. A scalar-producing node
 3. A bundle_means node that sets up the mapping

Here is a small example where we use autotrack bundles:

.. code-block:: yaml

  nodes:
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
