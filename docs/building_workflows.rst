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
an MRtrix3 ``csd`` node can load precomputed response functions,
such as a group-average response function shared by every participant in a study.
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

Multi-tissue FOD algorithms (``msmt_csd`` and ``ss3t``) need all three of
``wm_txt``, ``gm_txt``, and ``csf_txt``.
Single-tissue CSD (``fod: {algorithm: csd}``) needs only ``wm_txt``.
*QSIRecon* checks this, and that the paths are absolute, when it builds the workflow.

.. note::
    Earlier versions of *QSIRecon* located these files with the ``--recon-spec-aux-files``
    command-line option.
    That option has been removed; write the full paths in the recon spec instead.


Why use a common response function?
-----------------------------------

The amplitude of a fiber orientation distribution (FOD) is expressed relative to the
response function used to estimate it.
If each participant's FODs are estimated with that participant's own response function,
FOD amplitudes are on a different scale for each participant,
so measures derived from them cannot be compared directly between participants.
This includes apparent fiber density :footcite:p:`raffelt2012afd`
and fixel-based measures :footcite:p:`raffelt2017fba`.
The same applies to connection densities from SIFT2 :footcite:p:`smith2015sift2`.
Using one set of response functions for every participant
(together with multi-tissue intensity normalization, ``mtnormalize: true``)
puts all participants' FODs, and the measures derived from them, in the same units.

:footcite:t:`smith2022quantitative` explain in detail how to make SIFT2-weighted connectomes
comparable across participants. In that framework, each participant's connection densities are
scaled by the SIFT2 proportionality coefficient (mu), which *QSIRecon* writes out for each
participant as ``*_model-sift2_mu.txt``. That scaling is only meaningful
when every participant's FODs were estimated with the same response functions.
The `MRtrix3 fixel-based analysis documentation
<https://mrtrix.readthedocs.io/en/latest/fixel_based_analysis/mt_fibre_density_cross-section.html>`_
makes the same point for fixel-based analysis.


Worked example: estimating, averaging, and applying a group-average response function
-------------------------------------------------------------------------------------

This example uses two built-in recon specs:

* ``mrtrix_multishell_msmt_noACT_estimate`` estimates multi-tissue response functions
  (``dwi2response dhollander`` :footcite:p:`dhollander2019response`) for each participant,
  and stops there.
* ``mrtrix_multishell_msmt_noACT_apply`` fits multi-shell multi-tissue CSD with the response
  functions in ``/responses/{wm,gm,csf}.txt``, then runs tractography with SIFT2 and builds
  connectivity matrices.

The commands below use Apptainer and assume the QSIPrep outputs are in ``inputs/qsiprep``,
under the current directory.

**Step 1: Estimate a response function for each participant.**

.. code-block:: bash

   apptainer run \
       --containall \
       --writable-tmpfs \
       -B "${PWD}" \
       qsirecon-latest.sif \
       "${PWD}/inputs/qsiprep" \
       "${PWD}/results/estimate" \
       participant \
       -w "${PWD}/work/estimate" \
       --recon-spec mrtrix_multishell_msmt_noACT_estimate \
       -v -v

The response functions are written as text files, one per tissue type, to
``results/estimate/derivatives/qsirecon-MRtrix3_act-None_response-subject/``.
For example::

   sub-01/dwi/sub-01_space-T1w_model-dhollander_label-WM_response.txt
   sub-01/dwi/sub-01_space-T1w_model-dhollander_label-GM_response.txt
   sub-01/dwi/sub-01_space-T1w_model-dhollander_label-CSF_response.txt

**Step 2: Average the response functions across participants.**

Use MRtrix3's ``responsemean``, which is included in the *QSIRecon* container,
to average each tissue's response functions:

.. code-block:: bash

   estimate_dir="${PWD}/results/estimate/derivatives/qsirecon-MRtrix3_act-None_response-subject"
   mkdir -p "${PWD}/responses"
   for label in WM GM CSF; do
       lower_label=$(echo "${label}" | tr '[:upper:]' '[:lower:]')
       apptainer exec \
           --containall \
           -B "${PWD}" \
           qsirecon-latest.sif \
           responsemean \
           $(find "${estimate_dir}" -name "*_label-${label}_response.txt" | sort) \
           "${PWD}/responses/${lower_label}.txt"
   done

This writes ``responses/wm.txt``, ``responses/gm.txt``, and ``responses/csf.txt``.
``responsemean`` can only average response functions with the same number of b-values,
and it assumes the b-values themselves are the same, so average scans with
different acquisition schemes separately.
The average does not have to include every participant:
the MRtrix3 documentation suggests leaving out participants whose response functions could not be
estimated reliably, or whose whole brain is affected by pathology.

**Step 3: Apply the averaged response functions.**

Mount the ``responses`` directory at ``/responses``, where the built-in apply spec looks for it:

.. code-block:: bash

   apptainer run \
       --containall \
       --writable-tmpfs \
       -B "${PWD}" \
       -B "${PWD}/responses":/responses \
       qsirecon-latest.sif \
       "${PWD}/inputs/qsiprep" \
       "${PWD}/results/apply" \
       participant \
       -w "${PWD}/work/apply" \
       --recon-spec mrtrix_multishell_msmt_noACT_apply \
       --atlases 4S156Parcels \
       -v -v

The apply spec builds connectivity matrices, so it needs at least one atlas
(see :ref:`connectivity_atlases`).

With Docker, the equivalent mount is ``-v "${PWD}/responses":/responses:ro``:

.. code-block:: bash

   docker run -ti --rm \
       -v "${PWD}/inputs/qsiprep":/data:ro \
       -v "${PWD}/results/apply":/out \
       -v "${PWD}/work/apply":/work \
       -v "${PWD}/responses":/responses:ro \
       pennlinc/qsirecon:latest \
       /data /out participant \
       -w /work \
       --recon-spec mrtrix_multishell_msmt_noACT_apply \
       --atlases 4S156Parcels \
       -v -v

The outputs are written to
``results/apply/derivatives/qsirecon-MRtrix3_act-None_response-common/``,
including copies of the response functions that were used
(``*_model-precomputed_label-*_response.txt``),
the FODs, the SIFT2 weights and mu, and the connectivity matrices.

To restrict either run to some participants, use ``--participant-label``.
Every participant you want to compare should be processed with the same averaged
response functions.


Writing your own estimate and apply recon specs
-----------------------------------------------

The built-in specs are a starting point. For example, you could use a different response
algorithm, single-shell 3-tissue CSD, ACT, or response functions stored somewhere else.

An *estimate* spec has a ``csd`` node with a ``response`` section and **no** ``fod`` section.
Without a ``fod`` section, *QSIRecon* estimates and saves the response functions and skips FOD
estimation:

.. code-block:: yaml

  name: my_estimate
  anatomical: []
  space: T1w
  nodes:
  -   action: csd
      input: qsirecon
      name: estimate_response_functions
      parameters:
          response:
              algorithm: dhollander
      qsirecon_suffix: MRtrix3_act-None_response-subject
      software: MRTrix3

An *apply* spec is any spec whose ``csd`` node uses ``algorithm: precomputed``,
with absolute paths to the averaged files (see the example at the top of this section).
Downstream nodes, such as tractography and connectivity, are added as usual.
To use the files where they already are, rather than at ``/responses``,
copy the built-in apply spec and change ``wm_txt``, ``gm_txt``, and ``csf_txt``.
In a container, those paths must still be paths *inside* the container,
so mount the files there.
Then pass the spec's file path to ``--recon-spec``.


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
      - fit_noddi
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

.. _scalars_to_parcels:

Mapping scalar data to atlases
------------------------------

Scalars can also be summarized within the parcels of each atlas given to ``--atlases``.
A ``parcellate_scalars`` node writes the mean, standard deviation and median of each
scalar in each parcel, along with each parcel's coverage by the brain mask:

.. code-block:: yaml

  -   action: parcellate_scalars
      input: qsirecon
      name: parcellate_scalars
      scalars_from:
      - dipy_dki
      software: qsirecon

.. _qc_scalars_in_workflows:

Summarizing QSIPrep QC maps
---------------------------

The voxelwise QC maps that QSIPrep writes, such as its noise and distortion maps, can be
used as scalars (see :ref:`qsiprep_qc_scalars`). Add a ``qsiprep_qc_scalars`` node and
list it in ``scalars_from``:

.. code-block:: yaml

  -   action: qsiprep_qc_scalars
      input: qsirecon
      name: qc_scalars
      qsirecon_suffix: QSIPrepQC
      software: qsirecon
  -   action: bundle_map
      input: autotrackgqi
      name: bundle_means
      scalars_from:
      - dipy_dki
      - qc_scalars
      software: qsirecon
  -   action: template_map
      input: qsirecon
      name: template_map_linear
      parameters:
          interpolation: Linear
      scalars_from:
      - qc_scalars
      software: qsirecon

``parameters: {include: [...]}`` selects the maps to use, from ``noise``, ``snr``,
``tsnr``, ``cnr``, ``jacobian``, ``sdc``, ``rank`` and ``rankdensity``. The default is all
of them except ``rank`` and ``rankdensity``. The QC maps vary smoothly, so map them to
template space with ``Linear`` interpolation rather than ``NearestNeighbor``.


**********
References
**********

.. footbibliography::
