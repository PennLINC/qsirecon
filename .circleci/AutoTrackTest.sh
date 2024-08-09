#!/bin/bash

cat << DOC

Reconstruction workflow tests
=============================

All supported reconstruction workflows get tested

This tests the following features:

Inputs:
-------

 - qsirecon single shell results (data/DSDTI_fmap)
 - qsirecon multi shell results (data/DSDTI_fmap)

DOC
set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} multishell_output
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json
export FS_LICENSE=${TESTDIR}/data/license.txt

# Test AutoTrack
TESTNAME=autotrack
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/multishell_output/qsirecon
QSIRECON_CMD=$(run_qsirecon_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

${QSIRECON_CMD} \
	 -w ${TEMPDIR} \
	 --recon-input ${BIDS_INPUT_DIR} \
	 --sloppy \
         --stop-on-first-crash \
	 --recon-spec dsi_studio_autotrack \
	 --recon-only \
	 -vv
