.. include:: links.rst

.. _connectivity_matrices:

#####################
Connectivity matrices
#####################

Instead of offering a bewildering number of options for constructing connectivity matrices,
``qsirecon`` will construct as many connectivity matrices as it can given the reconstruction
methods.
It is **highly** recommended that you pick a weighting scheme before you run
these pipelines and only look at those numbers.
If you look at more than one weighting method be sure to adjust your statistics for the
additional comparisons.

To skip this step in your workflow, you can modify an existing recon pipeline by removing the
``action: connectivity`` section from the yaml file.

.. _connectivity_atlases:

*******
Atlases
*******

The following atlases are included in ``qsirecon``.
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

The ``qsirecon`` atlas set can be downloaded directly from
`box  <https://upenn.box.com/shared/static/8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz>`_.

The 4S atlas combines the Schaefer 2018 cortical atlas (version v0143) :footcite:p:`Schaefer_2017`
at 10 different resolutions (100, 200, 300, 400, 500, 600, 700, 800, 900, and 1000 parcels) with
the CIT168 subcortical atlas :footcite:p:`pauli2018high`,
the Diedrichson cerebellar atlas :footcite:p:`king2019functional`,
the HCP thalamic atlas :footcite:p:`najdenovska2018vivo`,
and the amygdala and hippocampus parcels from the HCP CIFTI subcortical parcellation
:footcite:p:`glasser2013minimal`.
The 4S atlas is used in the same manner across three PennLINC BIDS Apps:
QSIRecon, XCP-D, and ASLPrep, to produce synchronized outputs across modalities.
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
                  sub-<label>_space-T1w_seg-<label>_dseg.nii.gz
                  sub-<label>_space-T1w_seg-<label>_dseg.mif.gz
                  sub-<label>_space-T1w_seg-<label>_dseg.json
                  sub-<label>_space-T1w_seg-<label>_dseg.tsv


.. _custom_atlases:

Using custom atlases
^^^^^^^^^^^^^^^^^^^^

It's possible to use your own atlases provided you organize the atlases into BIDS-Atlas datasets.
Users can control which atlases are used with the ``--atlases`` and ``--datasets`` parameters.

The nifti images should be registered to the
`MNI152NLin2009cAsym <https://github.com/PennLINC/qsirecon/blob/main/qsirecon/data/mni_1mm_t1w_lps.nii.gz>`_
included in ``qsirecon``.
It is essential that your images are in the LPS+ orientation and have the sform zeroed-out in the header.
**Be sure to check for alignment and orientation** in your outputs.


*********************
Connectivity Measures
*********************

Connectivity measures are bundled together in binary ``.mat`` files,
rather than as atlas- and measure-specific tabular files.

.. warning::

   We ultimately plan to organize the connectivity matrices accoring to the BIDS-Connectivity BEP,
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


**********
References
**********

.. footbibliography::
