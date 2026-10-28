# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for the ABCD Study implementation of Restriction Spectrum Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is fit by running the unmodified ``RSIproc`` script from
https://github.com/ABCD-STUDY/RSI, which is vendored in ``qsirecon/data/rsi``.

"""

import os.path as op
import subprocess
import sys

import numpy as np
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from qsirecon.data import load as load_data

from ..utils.boilerplate import ConditionalDoc

LOGGER = logging.getLogger('nipype.interface')

RSI_VERSION = '1.0.8'
RSI_COMMIT = 'f775121a4ca9a0d8e9e355c60ce85b4fd7d879ac'
RSI_URL = f'https://github.com/ABCD-STUDY/RSI/tree/{RSI_COMMIT}'
RSI_SCRIPT = 'rsi/RSIproc_{}.py'.format(RSI_VERSION.replace('.', '_'))

# These mirror the constants hard-coded in the vendored RSIproc script.
# They are only used to describe the fit in the output metadata.
RSI_LAMBDA = 0.1
RSI_ADC_LONG = [1.0e-3, 1.0e-3, 3.0e-3]
RSI_ADC_TRANS = [1.0e-10, 0.9e-3, 3.0e-3]
RSI_SH_ORDER = [4, 4, 0]

# Names of the three RSIproc compartments (s1, s2 and s3)
RSI_COMPARTMENTS = ['restricted', 'hindered', 'free']

# RSIproc treats volumes with b <= 10 as b=0 volumes
RSI_B0_MAX = 10

# Maps the output trait names to the file names written by RSIproc
RSI_OUTPUTS = {
    'restrictediso': 'n0s1.nii.gz',
    'restricteddir': 'nds1.nii.gz',
    'restrictedtotal': 'nts1.nii.gz',
    'hinderediso': 'n0s2.nii.gz',
    'hindereddir': 'nds2.nii.gz',
    'hinderedtotal': 'nts2.nii.gz',
    'freeiso': 'n0s3.nii.gz',
    'freetotal': 'nts3.nii.gz',
    'l2norm': 'L2norm.nii.gz',
}


def _compartment_metadata():
    """Describe the RSIproc compartments with BEP016 model parameter names."""
    compartments = {}
    for name, d_long, d_trans, sh_degree in zip(
        RSI_COMPARTMENTS, RSI_ADC_LONG, RSI_ADC_TRANS, RSI_SH_ORDER
    ):
        if sh_degree == 0 and d_long == d_trans:
            compartments[name] = {'IsotropicDiffusivity': d_long}
        else:
            compartments[name] = {
                'ParallelDiffusivity': d_long,
                'PerpendicularDiffusivity': d_trans,
                'SphericalHarmonicsDegree': sh_degree,
            }

    return compartments


class _RSIInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='Preprocessed DWI series.')
    bval_file = File(exists=True, mandatory=True, desc='FSL-style bval file.')
    bvec_file = File(exists=True, mandatory=True, desc='FSL-style bvec file.')
    mask_file = File(exists=True, mandatory=True, desc='Brain mask for the DWI series.')
    b0_threshold = traits.Float(
        50.0,
        usedefault=True,
        desc=(
            'Threshold below which a b-value is considered a b0. '
            'These b-values are set to 0 before running RSIproc, '
            'which only treats b-values of 10 or less as b0.'
        ),
        doc=ConditionalDoc('All b-values below {value} s/mm^2 were treated as b=0.'),
    )


class _RSIOutputSpec(TraitedSpec):
    restrictediso = File(desc='Restricted isotropic signal component (RSIproc n0s1).')
    restrictediso_metadata = traits.Dict()
    restricteddir = File(desc='Restricted directional signal component (RSIproc nds1).')
    restricteddir_metadata = traits.Dict()
    restrictedtotal = File(desc='Restricted total signal component (RSIproc nts1).')
    restrictedtotal_metadata = traits.Dict()
    hinderediso = File(desc='Hindered isotropic signal component (RSIproc n0s2).')
    hinderediso_metadata = traits.Dict()
    hindereddir = File(desc='Hindered directional signal component (RSIproc nds2).')
    hindereddir_metadata = traits.Dict()
    hinderedtotal = File(desc='Hindered total signal component (RSIproc nts2).')
    hinderedtotal_metadata = traits.Dict()
    freeiso = File(desc='Free water isotropic signal component (RSIproc n0s3).')
    freeiso_metadata = traits.Dict()
    freetotal = File(desc='Free water total signal component (RSIproc nts3).')
    freetotal_metadata = traits.Dict()
    l2norm = File(desc='L2 norm of all model coefficients.')
    l2norm_metadata = traits.Dict()


class RSI(SimpleInterface):
    """Fit the restriction spectrum imaging (RSI) model with the ABCD Study code."""

    input_spec = _RSIInputSpec
    output_spec = _RSIOutputSpec

    def _run_interface(self, runtime):
        bvals = np.loadtxt(self.inputs.bval_file, ndmin=1)
        b0_mask = bvals < self.inputs.b0_threshold
        if not b0_mask.any():
            raise ValueError(
                f'RSI requires at least one b=0 image, but no b-values in '
                f'{self.inputs.bval_file} are below {self.inputs.b0_threshold}.'
            )

        # RSIproc only considers b <= 10 to be a b=0, so zero out low b-values
        bval_file = self.inputs.bval_file
        if np.any(bvals[b0_mask] > RSI_B0_MAX):
            bvals[b0_mask] = 0
            bval_file = fname_presuffix(
                self.inputs.bval_file,
                suffix='_rsi',
                newpath=runtime.cwd,
            )
            np.savetxt(bval_file, bvals[None, :], fmt='%g')

        # RSIproc writes all its outputs to the current working directory
        cmd = [
            sys.executable,
            str(load_data(RSI_SCRIPT)),
            self.inputs.dwi_file,
            self.inputs.mask_file,
            bval_file,
            self.inputs.bvec_file,
        ]
        LOGGER.info('Running RSI: %s', ' '.join(cmd))
        proc = subprocess.run(cmd, cwd=runtime.cwd, capture_output=True, text=True, check=False)
        runtime.stdout = proc.stdout
        runtime.stderr = proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(
                f'RSIproc exited with code {proc.returncode}.\n'
                f'Standard output:\n{proc.stdout}\n'
                f'Standard error:\n{proc.stderr}'
            )

        base_metadata = {
            'Model': {
                'Parameters': {
                    'Compartments': _compartment_metadata(),
                    'Lambda': RSI_LAMBDA,
                    'b0_threshold': self.inputs.b0_threshold,
                },
            },
            'Software': {
                'Name': 'RSIproc',
                'Version': RSI_VERSION,
                'URL': RSI_URL,
            },
        }
        for output_name, output_file in RSI_OUTPUTS.items():
            output_path = op.join(runtime.cwd, output_file)
            if not op.exists(output_path):
                raise FileNotFoundError(f'RSIproc did not produce {output_path}')

            self._results[output_name] = output_path
            self._results[f'{output_name}_metadata'] = base_metadata

        return runtime
