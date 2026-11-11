# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interface for running a PyAFQ workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import os
import os.path as op

from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import ImageFile, PVEImages
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')

# Recon spec parameters that QSIRecon always sets from its own inputs.
_INPUT_DERIVED_PARAMS = ('brain_mask_definition', 'import_tract', 'pve')


class PyAFQInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    t1_file = File(
        exists=True,
        mandatory=True,
        desc='T1w image registered to, but not resampled onto, the DWI',
    )
    mask_file = File(exists=True, mandatory=True, desc='brain mask on the T1w grid')
    csf_probseg = traits.Either(
        None,
        File(exists=True),
        desc='CSF probability map on the T1w grid',
    )
    gm_probseg = traits.Either(
        None,
        File(exists=True),
        desc='GM probability map on the T1w grid',
    )
    wm_probseg = traits.Either(
        None,
        File(exists=True),
        desc='WM probability map on the T1w grid',
    )
    kwargs = traits.Dict(exists=True, mandatory=True)
    tck_file = traits.Either(None, File(exists=True))
    n_procs = traits.Int(1, usedefault=True)
    export = traits.Str(
        'all',
        usedefault=True,
        desc='name of a single pyAFQ output to export, or "all" to export everything',
    )


class PyAFQOutputSpec(TraitedSpec):
    afq_dir = traits.Directory()


class PyAFQRecon(SimpleInterface):
    input_spec = PyAFQInputSpec
    output_spec = PyAFQOutputSpec

    def _run_interface(self, runtime):
        # shim the expected inputs
        shim_dir = op.join(runtime.cwd, 'study/subject')
        os.makedirs(shim_dir)
        files = {}
        for name in (
            'bval_file',
            'bvec_file',
            'dwi_file',
            't1_file',
            'mask_file',
            'csf_probseg',
            'gm_probseg',
            'wm_probseg',
            'tck_file',
        ):
            source = getattr(self.inputs, name)
            if not isdefined(source) or source is None:
                files[name] = None
                continue

            files[name] = fname_presuffix(source, newpath=shim_dir)
            os.symlink(source, files[name])

        kwargs = dict(self.inputs.kwargs)
        for key in _INPUT_DERIVED_PARAMS:
            if kwargs.pop(key, None) is not None:
                LOGGER.warning(
                    f'Ignoring "{key}" from the recon spec. QSIRecon sets it from its own inputs.'
                )

        # Tractography in pyAFQ is anatomically constrained, which requires CSF, GM and WM
        # probability maps. Use the ones that come with the preprocessed data instead of
        # letting pyAFQ segment the T1w with a neural network.
        probsegs = {name: files[f'{name.lower()}_probseg'] for name in ('CSF', 'GM', 'WM')}
        if all(probsegs.values()):
            kwargs['pve'] = PVEImages(
                **{f'{name}_probseg': ImageFile(path=path) for name, path in probsegs.items()}
            )
        elif files['tck_file'] is None:
            raise ValueError(
                'pyAFQ tractography requires CSF, GM and WM probability maps aligned with the '
                'T1w image (the QSIPrep label-{CSF,GM,WM}_probseg outputs), but at least one '
                'is missing. Provide them with the preprocessed data, or use a recon spec that '
                'imports tractography from another node (use_external_tracking: true).'
            )

        kwargs['brain_mask_definition'] = ImageFile(path=files['mask_file'])
        # pyAFQ runs tractography in numba threads
        kwargs['numba_n_threads'] = self.inputs.n_procs

        output_dir = shim_dir + '/PYAFQ/'
        os.makedirs(output_dir, exist_ok=True)
        myafq = ParticipantAFQ(
            files['dwi_file'],
            files['bval_file'],
            files['bvec_file'],
            files['t1_file'],
            output_dir,
            import_tract=files['tck_file'],
            **kwargs,
        )

        if self.inputs.export == 'all':
            myafq.export_all()
        else:
            myafq.export(self.inputs.export)

        self._results['afq_dir'] = output_dir

        return runtime
