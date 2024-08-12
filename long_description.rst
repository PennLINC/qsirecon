QSIRecon borrows heavily from fMRIPrep to build workflows for reconstructing q-space images
such as Diffusion Spectrum Images (DSI), multi-shell HARDI and compressed sensing DSI (CS-DSI).
QSIRecon can reconstruct orientation distribution functions (ODFs), fiber orientation
distributions (FODs) and perform tractography, estimate anisotropy scalars and connectivity
estimation using a combination of Dipy, MRTrix and DSI Studio using a JSON-based pipeline
specification.

Since QSIRecon uses the fMRIPrep workflow-building strategy,
it can also generate methods boilerplate and quality-check figures.

[Documentation `qsirecon.org <https://qsirecon.readthedocs.io>`_]
