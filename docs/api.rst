.. include:: links.rst

################
Developers - API
################

*****************************
Internal configuration system
*****************************

.. automodule:: qsirecon.config
   :members: from_dict, load, get, dumps, to_filename, init_spaces


***********
Library API
***********

Preprocessing Workflows
-----------------------

.. toctree::
   :glob:

   api/qsirecon.workflows.base
   api/qsirecon.workflows.anatomical
   api/qsirecon.workflows.dwi
   api/qsirecon.workflows.fieldmap


Reconstruction Workflows
------------------------

.. toctree::
   :glob:

   api/qsirecon.workflows.recon


Other Utilities
---------------

.. toctree::
   :glob:

   api/qsirecon.interfaces
   api/qsirecon.utils
   api/qsirecon.report
   api/qsirecon.viz
   api/qsirecon.qc
