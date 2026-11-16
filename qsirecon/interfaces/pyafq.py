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


class PyAFQBaseInputSpec(BaseInterfaceInputSpec):
    """Inputs that QSIRecon passes to a pyAFQ ParticipantAFQ."""

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
    n_procs = traits.Int(1, usedefault=True)


class PyAFQInputSpec(PyAFQBaseInputSpec):
    tck_file = traits.Either(None, File(exists=True))
    export = traits.Str(
        'all',
        usedefault=True,
        desc='name of a single pyAFQ output to export, or "all" to export everything',
    )


class PyAFQOutputSpec(TraitedSpec):
    afq_dir = traits.Directory()


def shim_pyafq_inputs(inputs, cwd, names):
    """Symlink the input files named in ``names`` into a pyAFQ-style study directory.

    Returns the directory and a dictionary mapping each name to its link,
    or to None for an input that is undefined or None.
    """
    shim_dir = op.join(cwd, 'study/subject')
    os.makedirs(shim_dir)
    files = {}
    for name in names:
        source = getattr(inputs, name)
        if not isdefined(source) or source is None:
            files[name] = None
            continue

        files[name] = fname_presuffix(source, newpath=shim_dir)
        os.symlink(source, files[name])

    return shim_dir, files


def participant_afq_kwargs(kwargs, files, n_procs):
    """Set the ParticipantAFQ arguments that QSIRecon derives from its own inputs.

    Parameters
    ----------
    kwargs : dict
        ParticipantAFQ keyword arguments from the recon spec. Not modified.
    files : dict
        Shimmed input files, from :func:`shim_pyafq_inputs`.
    n_procs : int
        Number of numba threads pyAFQ may use.

    Returns
    -------
    kwargs : dict
        The keyword arguments, with the brain mask and, when all three tissue probability
        maps are available, the tissue partial volume estimates set from ``files``.
    """
    kwargs = dict(kwargs)
    for key in _INPUT_DERIVED_PARAMS:
        if kwargs.pop(key, None) is not None:
            LOGGER.warning(
                f'Ignoring "{key}" from the recon spec. QSIRecon sets it from its own inputs.'
            )

    # Use the tissue probability maps that come with the preprocessed data instead of
    # letting pyAFQ segment the T1w with a neural network.
    probsegs = {name: files.get(f'{name.lower()}_probseg') for name in ('CSF', 'GM', 'WM')}
    if all(probsegs.values()):
        kwargs['pve'] = PVEImages(
            **{f'{name}_probseg': ImageFile(path=path) for name, path in probsegs.items()}
        )

    kwargs['brain_mask_definition'] = ImageFile(path=files['mask_file'])
    # pyAFQ runs tractography in numba threads
    kwargs['numba_n_threads'] = n_procs
    return kwargs


class PyAFQRecon(SimpleInterface):
    input_spec = PyAFQInputSpec
    output_spec = PyAFQOutputSpec

    def _run_interface(self, runtime):
        shim_dir, files = shim_pyafq_inputs(
            self.inputs,
            runtime.cwd,
            (
                'bval_file',
                'bvec_file',
                'dwi_file',
                't1_file',
                'mask_file',
                'csf_probseg',
                'gm_probseg',
                'wm_probseg',
                'tck_file',
            ),
        )
        kwargs = participant_afq_kwargs(self.inputs.kwargs, files, self.inputs.n_procs)

        # Tractography in pyAFQ is anatomically constrained, which requires CSF, GM and WM
        # probability maps.
        if 'pve' not in kwargs and files['tck_file'] is None:
            raise ValueError(
                'pyAFQ tractography requires CSF, GM and WM probability maps aligned with the '
                'T1w image (the QSIPrep label-{CSF,GM,WM}_probseg outputs), but at least one '
                'is missing. Provide them with the preprocessed data, or use a recon spec that '
                'imports tractography from another node (use_external_tracking: true).'
            )

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
