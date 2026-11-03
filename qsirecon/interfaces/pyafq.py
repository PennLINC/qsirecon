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


class PyAFQInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    t1_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    csf_probseg = traits.Either(None, File(exists=True))
    gm_probseg = traits.Either(None, File(exists=True))
    wm_probseg = traits.Either(None, File(exists=True))
    itk_file = File(exists=True, mandatory=True)
    kwargs = traits.Dict(exists=True, mandatory=True)
    tck_file = traits.Either(None, File(exists=True))
    n_procs = traits.Int(1, usedefault=True)


class PyAFQOutputSpec(TraitedSpec):
    afq_dir = traits.Directory()


class PyAFQRecon(SimpleInterface):
    input_spec = PyAFQInputSpec
    output_spec = PyAFQOutputSpec

    def _run_interface(self, runtime):
        # shim the expected inputs
        shim_dir = op.join(runtime.cwd, 'study/subject')
        os.makedirs(shim_dir)
        bval_file = fname_presuffix(self.inputs.bval_file, newpath=shim_dir)
        bvec_file = fname_presuffix(self.inputs.bvec_file, newpath=shim_dir)
        dwi_file = fname_presuffix(self.inputs.dwi_file, newpath=shim_dir)
        t1_file = fname_presuffix(self.inputs.t1_file, newpath=shim_dir)
        mask_file = fname_presuffix(self.inputs.mask_file, newpath=shim_dir)
        itk_file = fname_presuffix(self.inputs.itk_file, newpath=shim_dir)
        os.symlink(self.inputs.bval_file, bval_file)
        os.symlink(self.inputs.bvec_file, bvec_file)
        os.symlink(self.inputs.dwi_file, dwi_file)
        os.symlink(self.inputs.t1_file, t1_file)
        os.symlink(self.inputs.mask_file, mask_file)
        os.symlink(self.inputs.itk_file, itk_file)

        kwargs = dict(self.inputs.kwargs)
        export = kwargs.pop('export', 'all')
        tck_file = kwargs.pop('import_tract', None)
        kwargs.pop('brain_mask_definition', None)

        if self.inputs.tck_file and isdefined(self.inputs.tck_file):
            tck_file = fname_presuffix(self.inputs.tck_file, newpath=shim_dir)
            os.symlink(self.inputs.tck_file, tck_file)

        # QSIPrep's probsegs replace SynthSeg. FreeSurfer-only inputs have none.
        probsegs = (self.inputs.csf_probseg, self.inputs.gm_probseg, self.inputs.wm_probseg)
        if all(isdefined(path) and path for path in probsegs):
            kwargs.setdefault('pve', PVEImages(*[ImageFile(path=path) for path in probsegs]))

        output_dir = shim_dir + '/PYAFQ/'
        os.makedirs(output_dir, exist_ok=True)
        myafq = ParticipantAFQ(
            dwi_file,
            bval_file,
            bvec_file,
            t1_file,
            output_dir,
            import_tract=tck_file,
            brain_mask_definition=ImageFile(path=mask_file),
            numba_n_threads=self.inputs.n_procs,
            **kwargs,
        )

        if export == 'all':
            myafq.export_all()
        else:
            myafq.export(export)

        self._results['afq_dir'] = output_dir

        return runtime
