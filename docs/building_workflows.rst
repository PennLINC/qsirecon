.. include:: links.rst

.. _building_workflows:

###############################
Custom Reconstruction Workflows
###############################

QSIRecon workflows are defined in ``json`` files. The :ref:`builtin_workflows`
can be found in the QSIRecon ``json`` format on
`github <https://github.com/PennLINC/qsirecon/tree/main/qsirecon/data/pipelines>`_,
and are a great place to get started with customizing parts of these workflows.

This format has a few key parts.

***********************
Pipeline-level metadata
***********************

At the root level of the Recon Spec there are

.. code-block:: json

  {
    "name": "dsistudio_pipeline",
    "anatomical": ["mrtrix_5tt_hsvs"],
    "atlases": ["schaefer100" "schaefer200", "schaefer400", "brainnetome246",
                "aicha384", "gordon333", "aal116"],
    "nodes": []
  }

The ``"name"`` element defines the name of the pipeline. There will be a directory in your working
directory with this name. The ``"anatomical"`` field lists additional things to compute
based on the T1w or T2w images in the input data. Currently accepted values are

  * ``"mrtrix_5tt_hsvs"``: A MRtrix_ 5tt segmentation based on :footcite:t:`smith2020hybrid`.
    This requires :ref:`include_freesurfer`.

The ``"atlases"`` field is a list of segmentations (:ref:`connectivity_atlases`) that will be used to create
:ref:``



**************
Pipeline nodes
**************



A *node* in the QSIRecon ``"nodes"`` list represents a unit of processing.
Unlike NiPype nodes, which represent a single call to a commandline program
or function, a node here represents a set of steps that accomplish a task.
For example, if you want to perform CSD there are other steps that should
happen before actually fitting spherical harmonics to the data.
The entry for this in the ``"nodes"`` list could look like:

.. code-block:: json

  {
    "name": "msmt_csd",
    "software": "MRTrix3",
    "action": "csd",
    "qsirecon_suffix": "MRtrix3_act-FAST",
    "parameters": {
      "mtnormalize": true,
      "response": {
        "algorithm": "dhollander"
      },
      "fod": {
        "algorithm": "msmt_csd",
        "max_sh": [8, 8, 8]
      }
    }
  },

QSIRecon figures out which software to use based on the values specified
for ``"software"`` and ``"action"``. The full list of recognized actions
for each software package can be found in
:meth:`qsirecon.workflows.build_workflow.workflow_from_spec`.
All nodes must have a ``name`` element, this serves
as an id for this node and is used for :ref:`passing_downstream`.


.. _connecting_nodes:

Connecting Nodes
================

Mixing between software packages is something QSIRecon generally does well.
There are a number of ways that nodes can exchange data with one another.

.. _passing_downstream:

Passing method-specific outputs from one node as inputs to another
------------------------------------------------------------------

When one node produces outputs that are specifically used as inputs for
another node, you can pass them by matching the ``"input"`` field to the
name of the upstream node. Here is an example connecting a CSD calculation
to MRtrix3 tractography.

.. code-block:: json

  {
    "name": "msmt_csd",
    "software": "MRTrix3",
    "action": "csd",
    "qsirecon_suffix": "MRtrix3_act-FAST"
  },
    {
      "name": "track_ifod2",
      "software": "MRTrix3",
      "action": "tractography",
      "qsirecon_suffix": "MRtrix3_act-FAST",
      "input": "msmt_csd"
    },

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

.. code-block:: json

  "nodes": [
    {
      "name": "dipy_dki",
      "software": "Dipy",
      "action": "DKI_reconstruction",
      "qsirecon_suffix": "DIPYDKI"
    },
    {
      "name": "fit_noddi",
      "action": "fit_noddi",
      "software": "AMICO",
      "qsirecon_suffix": "NODDI"
    },
    {
      "name": "template_map",
      "software": "qsirecon",
      "action": "template_map",
      "input": "qsirecon",
      "scalars_from": [
        "dipy_dki",
        "fit_noddi"
      ],
      "parameters": {
        "interpolation": "NearestNeighbor"
      }
    }
  ]

By listing the names of the scalar-producing nodes in the ``"scalars_from"`` field
you will end up with the scalars in both subject native and template space in the
output directory for each node that produces the scalars.

.. _scalars_to_bundles:

Mapping scelar data to bundles
------------------------------

Perhaps the most biologically meaningful unit of analysis for dMRI is
the bundle. Much like :ref:`scalars_resampling`, the scalar maps
produced by nodes can be summarized along bundles. The requirements
for bundle mapping are

 1. A bundle-creating node (such as autotrack or pyafq)
 2. A scalar-producing node
 3. A bundle_means node that sets up the mapping

Here is a small example where we use autotrack bundles:

.. code-block:: json

  "nodes": [
    {
      "name": "dipy_dki",
      "software": "Dipy",
      "action": "DKI_reconstruction",
      "qsirecon_suffix": "DIPYDKI"
    },
    {
      "name": "fit_noddi",
      "action": "fit_noddi",
      "software": "AMICO",
      "qsirecon_suffix": "NODDI"
    },
    {
      "name": "autotrackgqi",
      "software": "DSI Studio",
      "action": "autotrack",
      "input": "dsistudio_gqi",
      "qsirecon_suffix": "DSIStudio"
    },
    {
      "name": "bundle_means",
      "software": "qsirecon",
      "action": "bundle_map",
      "input": "autotrackgqi",
      "scalars_from": [
        "dipy_dki",
        "fit_noddi"
      ]
    }
  ]

This will produce a tsv with the NODDI and DKI scalars summarized for
each bundle produced by autotrack.


**********
References
**********

.. footbibliography::
