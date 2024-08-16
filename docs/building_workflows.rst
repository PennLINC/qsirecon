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

  * ``"mrtrix_5tt_hsvs"``: A MRtrix_ 5tt segmentation based on [Smith2020]_. This requires :ref:`include_freesurfer`

The ``"atlases"`` field is a list of segmentations (:ref:`connectivity_atlases`) that will be used to create
:ref:``



**************
Pipeline nodes
**************


The ``"nodes"`` list contains the workflows that will be run as a part of the
reconstruction pipeline. All nodes must have a ``name`` element, this serves
as an id for this node and is used to connect its outputs to a downstream
node. In this example we can see that the node with ``"name": "dsistudio_gqi"``
sends its outputs to the node with ``"name": "scalar_export"`` because
the ``"name": "scalar_export"`` node specifies ``"input": "dsistudio_gqi"``.
If no ``"input"`` is specified for a node, it is assumed that the
outputs from ``qsirecon`` will be its inputs.

By specifying ``"software": "DSI Studio"`` we will be using algorithms implemented
in `DSI Studio`_. Other options include MRTrix_ and Dipy_. Since there are many
things that `DSI Studio`_ can do, we specify that we want to reconstruct the
output from ``qsirecon`` by adding ``"action": "reconstruction"``. Additional
parameters can be sent to specify how the reconstruction should take place in
the ``"parameters"`` item. Possible options for ``"software"``, ``"action"``
and ``"parameters"`` can be found in the :ref:`builtin_reconstruction` section.

You will have access to all the intermediate data in the pipeline's working directory,
but can specify which outputs you want to save to the output directory by setting
an ``"output_suffix"``. Looking at the outputs for a workflow in the :ref:`builtin_reconstruction`
section you can see what is produced by each workflow. Each of these files
will be saved in your output directory for each subject with a name matching
your specified ``"output_suffix"``. In this case it will produce a file
``something_space-T1w_gqi.fib.gz``.  Since a fib file is produced by this node
and the downstream ``export_scalars`` node uses it, the scalars produced from
that node will be from this same fib file.

