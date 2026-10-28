# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for Restriction Spectrum Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is fit with :mod:`qsirecon.utils.rsi`, a port of the ABCD Study's ``RSIproc``
(https://github.com/ABCD-STUDY/RSI) that exposes its model settings.
The defaults reproduce RSIproc.

"""

import nibabel as nb
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

from ..utils.boilerplate import ConditionalDoc
from ..utils.rsi import (
    RSI_ADC_LONG,
    RSI_ADC_TRANS,
    RSI_COMPARTMENTS,
    RSI_LAMBDA,
    RSI_SH_ORDER,
    RSI_URL,
    RSI_VERSION,
    fit_rsi,
)

LOGGER = logging.getLogger('nipype.interface')

# Maps each output to the fit_rsi result and compartment index it comes from,
# and to the name of the corresponding RSIproc output
RSI_OUTPUTS = {
    'restrictediso': ('iso', 0, 'n0s1'),
    'restricteddir': ('dir', 0, 'nds1'),
    'restrictedtotal': ('total', 0, 'nts1'),
    'hinderediso': ('iso', 1, 'n0s2'),
    'hindereddir': ('dir', 1, 'nds2'),
    'hinderedtotal': ('total', 1, 'nts2'),
    'freeiso': ('iso', 2, 'n0s3'),
    'freetotal': ('total', 2, 'nts3'),
    'l2norm': ('l2norm', None, 'L2norm'),
}

# Spherical harmonic degrees allowed for the anisotropic compartments
SH_DEGREES = (2, 4, 6, 8, 10, 12)


class _RSIInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='Preprocessed DWI series.')
    bval_file = File(exists=True, mandatory=True, desc='FSL-style bval file.')
    bvec_file = File(exists=True, mandatory=True, desc='FSL-style bvec file.')
    mask_file = File(exists=True, mandatory=True, desc='Brain mask for the DWI series.')
    restricted_parallel_diffusivity = traits.Float(
        RSI_ADC_LONG[0],
        usedefault=True,
        desc='Diffusivity of the restricted compartment parallel to the fiber, in mm^2/s.',
    )
    restricted_perpendicular_diffusivity = traits.Float(
        RSI_ADC_TRANS[0],
        usedefault=True,
        desc='Diffusivity of the restricted compartment perpendicular to the fiber, in mm^2/s.',
    )
    restricted_sh_degree = traits.Enum(
        RSI_SH_ORDER[0],
        *SH_DEGREES,
        usedefault=True,
        desc='Maximum spherical harmonic degree of the restricted compartment.',
    )
    hindered_parallel_diffusivity = traits.Float(
        RSI_ADC_LONG[1],
        usedefault=True,
        desc='Diffusivity of the hindered compartment parallel to the fiber, in mm^2/s.',
    )
    hindered_perpendicular_diffusivity = traits.Float(
        RSI_ADC_TRANS[1],
        usedefault=True,
        desc='Diffusivity of the hindered compartment perpendicular to the fiber, in mm^2/s.',
    )
    hindered_sh_degree = traits.Enum(
        RSI_SH_ORDER[1],
        *SH_DEGREES,
        usedefault=True,
        desc='Maximum spherical harmonic degree of the hindered compartment.',
    )
    free_diffusivity = traits.Float(
        RSI_ADC_LONG[2],
        usedefault=True,
        desc='Diffusivity of the isotropic free water compartment, in mm^2/s.',
    )
    regularization_lambda = traits.Float(
        RSI_LAMBDA,
        usedefault=True,
        desc='Weight of the Tikhonov regularization (lambda).',
    )
    b0_threshold = traits.Float(
        50.0,
        usedefault=True,
        desc='Threshold below which a b-value is considered a b0.',
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


def rsi_model_settings(inputs):
    """Return the parallel and perpendicular diffusivities and SH degrees of each compartment."""
    parallel = [
        inputs.restricted_parallel_diffusivity,
        inputs.hindered_parallel_diffusivity,
        inputs.free_diffusivity,
    ]
    perpendicular = [
        inputs.restricted_perpendicular_diffusivity,
        inputs.hindered_perpendicular_diffusivity,
        inputs.free_diffusivity,
    ]
    sh_degrees = [inputs.restricted_sh_degree, inputs.hindered_sh_degree, 0]
    return parallel, perpendicular, sh_degrees


def describe_rsi_model(inputs):
    """Describe the RSI model settings for the methods boilerplate."""
    parallel, perpendicular, sh_degrees = rsi_model_settings(inputs)
    if sh_degrees[0] == sh_degrees[1]:
        sh_degree_str = f'up to degree {sh_degrees[0]}'
    else:
        sh_degree_str = f'up to degree {sh_degrees[0]} and {sh_degrees[1]}, respectively'

    return (
        'The diffusion-weighted signal, normalized to the mean b=0 signal, was modeled as '
        f'the sum of a restricted compartment (parallel diffusivity {parallel[0]:g} mm^2/s, '
        f'perpendicular diffusivity {perpendicular[0]:g} mm^2/s), a hindered compartment '
        f'(parallel diffusivity {parallel[1]:g} mm^2/s, perpendicular diffusivity '
        f'{perpendicular[1]:g} mm^2/s) and a free water compartment (isotropic diffusivity '
        f'{parallel[2]:g} mm^2/s). '
        'Fiber orientation distributions of the restricted and hindered compartments were '
        f'represented with spherical harmonics {sh_degree_str}, and the model was fit with '
        f'Tikhonov-regularized linear least squares (lambda = {inputs.regularization_lambda:g}). '
    )


def _compartment_metadata(inputs):
    """Describe the RSI compartments with BEP016 model parameter names."""
    parallel, perpendicular, sh_degrees = rsi_model_settings(inputs)
    compartments = {}
    for name, d_par, d_perp, sh_degree in zip(
        RSI_COMPARTMENTS, parallel, perpendicular, sh_degrees
    ):
        if sh_degree == 0:
            compartments[name] = {'IsotropicDiffusivity': d_par}
        else:
            compartments[name] = {
                'ParallelDiffusivity': d_par,
                'PerpendicularDiffusivity': d_perp,
                'SphericalHarmonicsDegree': sh_degree,
            }

    return compartments


class RSI(SimpleInterface):
    """Fit the restriction spectrum imaging (RSI) model."""

    input_spec = _RSIInputSpec
    output_spec = _RSIOutputSpec

    def _run_interface(self, runtime):
        dwi_img = nb.load(self.inputs.dwi_file)
        mask_img = nb.load(self.inputs.mask_file)
        bvals = np.loadtxt(self.inputs.bval_file, ndmin=1)
        bvecs = np.loadtxt(self.inputs.bvec_file, ndmin=2).T
        parallel, perpendicular, sh_degrees = rsi_model_settings(self.inputs)

        LOGGER.info('Fitting RSI model.')
        results = fit_rsi(
            np.asanyarray(dwi_img.dataobj),
            np.asanyarray(mask_img.dataobj),
            bvals,
            bvecs,
            parallel_diffusivities=parallel,
            perpendicular_diffusivities=perpendicular,
            sh_degrees=sh_degrees,
            regularization_lambda=self.inputs.regularization_lambda,
            b0_threshold=self.inputs.b0_threshold,
        )

        base_metadata = {
            'Model': {
                'Parameters': {
                    'Compartments': _compartment_metadata(self.inputs),
                    'Lambda': self.inputs.regularization_lambda,
                    'b0_threshold': self.inputs.b0_threshold,
                },
            },
            'Software': {
                'Name': 'QSIRecon port of RSIproc',
                'Version': RSI_VERSION,
                'URL': RSI_URL,
            },
        }
        header = mask_img.header.copy()
        header.set_data_dtype(np.float32)
        for output_name, (result_name, i_comp, _) in RSI_OUTPUTS.items():
            data = results[result_name]
            if i_comp is not None:
                data = data[i_comp]

            output_file = fname_presuffix(
                self.inputs.dwi_file,
                suffix=f'_{output_name}.nii.gz',
                newpath=runtime.cwd,
                use_ext=False,
            )
            nb.Nifti1Image(data.astype(np.float32), mask_img.affine, header).to_filename(
                output_file
            )
            self._results[output_name] = output_file
            self._results[f'{output_name}_metadata'] = base_metadata

        return runtime
