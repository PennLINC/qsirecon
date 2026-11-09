# ABCD Study Restriction Spectrum Imaging (RSI)

The files in this directory were copied without modification from
https://github.com/ABCD-STUDY/RSI at commit
`f775121a4ca9a0d8e9e355c60ce85b4fd7d879ac`.
They are distributed under the MIT License in `LICENSE`.

QSIRecon fits RSI with `qsirecon/utils/rsi.py`, a port of `RSIproc_1_0_8.py`
that exposes the model settings this script hard-codes.
These files are the reference implementation for that port:
`qsirecon/tests/test_rsi.py` runs `RSIproc_1_0_8.py` and checks that the port
reproduces its outputs, both with the default settings and with the script's
constants replaced by custom settings.

Do not edit these files.
To update to a new RSIproc release, copy the new upstream files here,
update the version, commit and default settings in `qsirecon/utils/rsi.py`,
and port any changes to the model until `test_rsi.py` passes.
