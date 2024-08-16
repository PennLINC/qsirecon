.. include:: links.rst

.. _connectivity_matrices:

#####################
Connectivity matrices
#####################

Instead of offering a bewildering number of options for constructing connectivity matrices,
``qsirecon`` will construct as many connectivity matrices as it can given the reconstruction
methods. It is **highly** recommended that you pick a weighting scheme before you run
these pipelines and only look at those numbers. If you look at more than one weighting method
be sure to adjust your statistics for the additional comparisons.

.. _connectivity_atlases:

*******
Atlases
*******

The following atlases are included in ``qsirecon`` and are used by default in the
:ref:`preconfigured_workflows`. If you use one of them please be sure to cite
the relevant publication.

 * ``schaefer100``, ``schaefer200``, ``schaefer400``: [Schaefer2017]_, [Yeo2011]_
 * ``brainnetome246``: [Fan2016]_
 * ``aicha384``: [Joliot2015]_
 * ``gordon333``: [Gordon2014]_
 * ``aal116``: [TzourioMazoyer2002]_

.. _custom_atlases:

Using custom atlases
^^^^^^^^^^^^^^^^^^^^

It's possible to use your own atlases provided you can match the format ``qsirecon`` uses to
read atlases. The ``qsirecon`` atlas set can be downloaded directly from
`box  <https://upenn.box.com/shared/static/8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz>`_.

In this directory there must exist a JSON file called ``atlas_config.json`` containing an
entry for each atlas you would like included. The format is::

  {
    "my_custom_atlas": {
      "file": "file_in_this_directory.nii.gz",
      "node_names": ["Region1_L", "Region1_R" ... "RegionN_R"],
      "node_ids": [1, 2, ..., N]
    }
    ...
  }

Where ``"node_names"`` are the text names of the regions in ``"my_custom_atlas"`` and
``"node_ids"`` are the numbers in the nifti file that correspond to each region. When
:ref:`custom_reconstruction` you can then inclued ``"my_custom_atlas"`` in the ``"atlases":[]``
section.

The directory containing ``atlas_config.json`` and the atlas nifti files should be mounted in
the container at ``/atlas/qsirecon_atlases``. If using ``qsirecon-docker`` or
``qsirecon-singularity`` this can be done with ``--custom-atlases /path/to/my/atlases`` or
if you're running on your own system (not recommended) you can set the environment variable
``QSIRECON_ATLAS=/path/to/my/atlases``.

The nifti images should be registered to the
`MNI152NLin2009cAsym <https://github.com/PennLINC/qsirecon/blob/main/qsirecon/data/mni_1mm_t1w_lps.nii.gz>`_
included in ``qsirecon``.
It is essential that your images are in the LPS+ orientation and have the sform zeroed-out
in the header. **Be sure to check for alignment and orientation** in your outputs.
