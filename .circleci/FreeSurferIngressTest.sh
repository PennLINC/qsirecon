#!/bin/bash

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
#get_bids_data ${TESTDIR} freesurfer
#get_bids_data ${TESTDIR} abcd_output

CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

# Test dipy_mapmri
TESTNAME=fs_ingress_test
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/araikes/qsirecon
SUBJECTS_DIR=${TESTDIR}/data/freesurfer
QSIRECON_CMD=$(run_qsirecon_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

${QSIRECON_CMD} \
	 -w ${TEMPDIR} \
	 --recon-input ${BIDS_INPUT_DIR} \
	 --sloppy \
	 --recon-spec ${PWD}/test_5tt_hsv.json \
	 --freesurfer-input ${SUBJECTS_DIR} \
	 --recon-only \
	 -vv --debug pdb



