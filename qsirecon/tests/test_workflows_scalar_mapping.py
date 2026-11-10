"""Tests for mapping scalars to the template spaces requested with --output-spaces."""

import nibabel as nb
import numpy as np
import pytest
from bids.layout import BIDSLayout
from nipype.interfaces.base import isdefined

from qsirecon import config
from qsirecon.interfaces.bids import DerivativesDataSink, get_recon_output_name
from qsirecon.interfaces.interchange import recon_workflow_input_fields
from qsirecon.tests.test_workflows_anatomical import _unconnected_mandatory_inputs
from qsirecon.workflows.recon.scalar_mapping import (
    _describe_template_mapping,
    init_scalar_to_template_wf,
)

DWI_NAME = 'sub-01_ses-1_space-ACPC_desc-preproc_dwi.nii.gz'


@pytest.fixture
def xfms(tmp_path):
    """Transform files for an adult and an infant template, as the build resolves them."""
    files = {}
    for name in (
        'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'sub-01_from-ACPC_to-MNIInfant+2_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ):
        (tmp_path / name).touch()
        files[name.split('_')[1 if name.startswith('tpl') else 2]] = str(tmp_path / name)

    return {
        'adult': [files['to-MNI152NLin2009cAsym']],
        'infant': [files['to-MNIInfant+2']],
        # A chain, in the order ANTs takes it
        'chained': [files['from-MNI152NLin2009cAsym'], files['to-MNI152NLin2009cAsym']],
    }


def _space_xfms(token, space, files, transform_path=None):
    return {
        'token': token,
        'space': space,
        'acpc_to_template_xfm': files,
        'transform_path': transform_path,
    }


def _build(tmp_path, monkeypatch, output_space_xfms):
    monkeypatch.setattr(config.execution, 'output_dir', tmp_path / 'out')
    return init_scalar_to_template_wf(
        inputs_dict={
            'dwi_file': str(tmp_path / DWI_NAME),
            'has_qsiprep_anat_transforms': True,
            'output_space_xfms': output_space_xfms,
        },
        name='template_map',
        qsirecon_suffix='DIPYDKI',
    )


@pytest.mark.parametrize(
    'inputs_dict',
    [
        {'has_qsiprep_anat_transforms': False, 'output_space_xfms': []},
        {'has_qsiprep_anat_transforms': False},
        # The flag alone is not enough: there must be a space to write in
        {'has_qsiprep_anat_transforms': True, 'output_space_xfms': []},
    ],
)
def test_scalar_to_template_wf_needs_template(inputs_dict):
    """Mapping scalars to a template fails clearly when there is no template space."""
    with pytest.raises(ValueError, match='requires ACPC-to-template transforms'):
        init_scalar_to_template_wf(inputs_dict=inputs_dict)


def test_scalar_to_template_wf_default_space(tmp_path, monkeypatch, xfms):
    """Without --output-spaces: one space, the voxel size of the DWI, and no res- entity."""
    wf = _build(
        tmp_path,
        monkeypatch,
        [_space_xfms('MNI152NLin2009cAsym', 'MNI152NLin2009cAsym', xfms['adult'])],
    )

    mapper = wf.get_node('template_mapper_MNI152NLin2009cAsym')
    assert mapper.inputs.template_space == 'MNI152NLin2009cAsym'
    assert mapper.inputs.to_template_transform == xfms['adult']

    get_template = wf.get_node('get_template_MNI152NLin2009cAsym_tfres_1')
    assert get_template.inputs.template_name == 'MNI152NLin2009cAsym'
    assert get_template.inputs.resolution == '1'

    # The voxel size is chosen from the DWI when the workflow runs
    grid_wf = wf.get_node('output_grid_MNI152NLin2009cAsym_wf')
    assert grid_wf.get_node('voxel_size_chooser') is not None
    assert not isdefined(grid_wf.get_node('resample_to_voxel_size').inputs.voxel_size)

    ds_scalar = wf.get_node('scalar_output_MNI152NLin2009cAsym_wf').get_node('ds_scalar')
    assert not isdefined(ds_scalar.inputs.res)
    assert ds_scalar.inputs.base_directory.endswith('qsirecon-DIPYDKI')

    assert len([name for name in wf.list_node_names() if 'template_mapper' in name]) == 1
    assert wf.__desc__ == (
        'Scalar NIfTI files were warped to MNI152NLin2009cAsym template space, '
        'at the smallest voxel dimension of the preprocessed DWI.'
    )


def test_scalar_to_template_wf_fans_out(tmp_path, monkeypatch, xfms):
    """One grid, mapper and set of datasinks is built per requested space and resolution."""
    chain = ['ACPC', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym']
    wf = _build(
        tmp_path,
        monkeypatch,
        [
            _space_xfms('MNI152NLin2009cAsym:res-2', 'MNI152NLin2009cAsym', xfms['adult']),
            _space_xfms('MNI152NLin2009cAsym:res-2mm', 'MNI152NLin2009cAsym', xfms['adult']),
            _space_xfms('MNI152NLin2009cAsym:res-1p5mm', 'MNI152NLin2009cAsym', xfms['adult']),
            _space_xfms('MNIInfant:cohort-2', 'MNIInfant+2', xfms['infant']),
            _space_xfms('MNI152NLin6Asym:res-1', 'MNI152NLin6Asym', xfms['chained'], chain),
        ],
    )

    # label: (space entity, transforms, res entity, voxel size set at build time)
    expected = {
        'MNI152NLin2009cAsym_res_2': ('MNI152NLin2009cAsym', xfms['adult'], '2', None),
        'MNI152NLin2009cAsym_res_2mm': ('MNI152NLin2009cAsym', xfms['adult'], '2mm', (2.0,) * 3),
        'MNI152NLin2009cAsym_res_1p5mm': (
            'MNI152NLin2009cAsym',
            xfms['adult'],
            '1p5mm',
            (1.5,) * 3,
        ),
        'MNIInfant_cohort_2': ('MNIInfant+2', xfms['infant'], None, None),
        'MNI152NLin6Asym_res_1': ('MNI152NLin6Asym', xfms['chained'], '1', None),
    }
    mappers = sorted(name for name in wf.list_node_names() if name.startswith('template_mapper'))
    assert mappers == sorted(f'template_mapper_{label}' for label in expected)

    for label, (space, transforms, res, voxel_size) in expected.items():
        mapper = wf.get_node(f'template_mapper_{label}')
        assert mapper.inputs.template_space == space
        assert mapper.inputs.to_template_transform == transforms

        ds_scalar = wf.get_node(f'scalar_output_{label}_wf').get_node('ds_scalar')
        if res is None:
            assert not isdefined(ds_scalar.inputs.res)
        else:
            assert ds_scalar.inputs.res == res

        assert ds_scalar.inputs.base_directory.endswith('qsirecon-DIPYDKI')

        grid_wf = wf.get_node(f'output_grid_{label}_wf')
        resample = grid_wf.get_node('resample_to_voxel_size')
        if voxel_size is not None:
            assert resample.inputs.voxel_size == voxel_size

    # A TemplateFlow label keeps the template's own grid
    for label in ('MNI152NLin2009cAsym_res_2', 'MNI152NLin6Asym_res_1'):
        grid_wf = wf.get_node(f'output_grid_{label}_wf')
        assert grid_wf.get_node('resample_to_voxel_size') is None

    # No resolution: the voxel size of the DWI, and the tighter field of view for infants
    infant_grid = wf.get_node('output_grid_MNIInfant_cohort_2_wf')
    assert infant_grid.get_node('voxel_size_chooser') is not None
    assert infant_grid.get_node('autobox_template').inputs.padding == 4
    adult_grid = wf.get_node('output_grid_MNI152NLin2009cAsym_res_2mm_wf')
    assert adult_grid.get_node('autobox_template').inputs.padding == 8

    # The template is fetched once per TemplateFlow resolution: the two physical sizes
    # both start from the res-1 image
    templates = {
        name: wf.get_node(name).inputs.get()
        for name in wf.list_node_names()
        if name.startswith('get_template')
    }
    assert {
        name: (inputs['template_name'], inputs['resolution']) for name, inputs in templates.items()
    } == {
        'get_template_MNI152NLin2009cAsym_tfres_2': ('MNI152NLin2009cAsym', '2'),
        'get_template_MNI152NLin2009cAsym_tfres_1': ('MNI152NLin2009cAsym', '1'),
        'get_template_MNIInfant_2_tfres_1': ('MNIInfant+2', '1'),
        'get_template_MNI152NLin6Asym_tfres_1': ('MNI152NLin6Asym', '1'),
    }

    # Every space's outputs are gathered on the outputnode
    assert wf.get_node('merge_template_scalars').interface._numinputs == len(expected)
    assert wf.get_node('merge_template_recon_scalars').interface._numinputs == len(expected)

    # Nothing is left unconnected, with only the usual workflow inputs supplied
    provided = [*recon_workflow_input_fields, 'recon_scalars', 'collected_scalars']
    assert _unconnected_mandatory_inputs(wf, provided) == []

    assert wf.__desc__ == (
        'Scalar NIfTI files were warped to 3 template spaces: '
        "MNI152NLin2009cAsym (on the template's res-2 grid and with 2 mm isotropic voxels "
        'and with 1.5 mm isotropic voxels); '
        'MNIInfant+2 (at the smallest voxel dimension of the preprocessed DWI); '
        "MNI152NLin6Asym (on the template's res-1 grid). "
        'To reach MNI152NLin6Asym, the transform from preprocessing to MNI152NLin2009cAsym '
        'was combined with template-to-template transforms from TemplateFlow.'
    )


def test_scalar_to_template_wf_does_not_use_removed_fields(tmp_path, monkeypatch, xfms):
    """The transforms and grids are set at build time, not passed as workflow fields."""
    for name in ('acpc_to_template_xfm', 'template_to_acpc_xfm', 'resampling_template'):
        assert name not in recon_workflow_input_fields

    wf = _build(
        tmp_path,
        monkeypatch,
        [_space_xfms('MNI152NLin2009cAsym:res-2mm', 'MNI152NLin2009cAsym', xfms['adult'])],
    )
    graph = wf._create_flat_graph()
    used = {
        src if isinstance(src, str) else src[0]
        for node, _, data in graph.out_edges(data=True)
        if node.name == 'inputnode' and node.fullname == 'template_map.inputnode'
        for src, _ in data['connect']
    }
    assert used == {'dwi_ref', 'dwi_file', 'collected_scalars'}


def test_describe_template_mapping_one_space_two_resolutions():
    """Resolutions of one template are described together."""
    desc = _describe_template_mapping(
        [
            _space_xfms('MNI152NLin6Asym:res-2', 'MNI152NLin6Asym', []),
            _space_xfms('MNI152NLin6Asym:res-1p5x1p5x3mm', 'MNI152NLin6Asym', []),
        ]
    )
    assert desc == (
        'Scalar NIfTI files were warped to MNI152NLin6Asym template space, '
        "on the template's res-2 grid and with 1.5 x 1.5 x 3 mm voxels."
    )


def _write_scalar(path):
    nb.Nifti1Image(np.zeros((2, 2, 2), dtype='float32'), np.eye(4)).to_filename(path)
    return str(path)


def _sink_scalar(tmp_path, source_name, **entities):
    """Write a scalar map the way ``init_scalar_output_wf`` does, and return its name."""
    in_file = _write_scalar(tmp_path / 'scalar.nii.gz')
    source_file = tmp_path / 'in' / 'sub-01' / 'ses-1' / 'dwi' / source_name
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.touch()

    sink = DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        dismiss_entities=['desc'],
        datatype='dwi',
        suffix='dwimap',
        extension='nii.gz',
        in_file=in_file,
        source_file=str(source_file),
        model='dki',
        param='fa',
        **entities,
    )
    result = sink.run(cwd=str(tmp_path))
    out_file = result.outputs.out_file
    assert '/sub-01/ses-1/dwi/' in out_file.replace('\\', '/')
    # The override of dismiss_entities does not outlive the run
    assert sink.inputs.dismiss_entities == ['desc']
    return out_file.replace('\\', '/').rsplit('/', 1)[-1]


def test_scalar_filenames_for_two_resolutions(tmp_path):
    """Two resolutions of one template get different names, through the res entity."""
    names = [
        _sink_scalar(tmp_path, DWI_NAME, space='MNI152NLin2009cAsym', res=res)
        for res in ('1', '2mm')
    ]
    assert names == [
        'sub-01_ses-1_space-MNI152NLin2009cAsym_res-1_model-dki_param-fa_dwimap.nii.gz',
        'sub-01_ses-1_space-MNI152NLin2009cAsym_res-2mm_model-dki_param-fa_dwimap.nii.gz',
    ]

    # No resolution was requested for the space: no res- entity
    assert _sink_scalar(tmp_path, DWI_NAME, space='MNI152NLin2009cAsym') == (
        'sub-01_ses-1_space-MNI152NLin2009cAsym_model-dki_param-fa_dwimap.nii.gz'
    )

    # An anisotropic size, after a cohort
    assert _sink_scalar(tmp_path, DWI_NAME, space='MNIInfant+2', res='1p5x1p5x3mm') == (
        'sub-01_ses-1_space-MNIInfant+2_res-1p5x1p5x3mm_model-dki_param-fa_dwimap.nii.gz'
    )


def test_scalar_filenames_never_inherit_res(tmp_path, monkeypatch):
    """A res- in the name of the input DWI is not passed on to any output."""
    source_name = 'sub-01_ses-1_space-ACPC_res-1p7mm_desc-preproc_dwi.nii.gz'

    # In the space of the DWI, as before the res entity was added to the path patterns
    assert _sink_scalar(tmp_path, source_name) == (
        'sub-01_ses-1_space-ACPC_model-dki_param-fa_dwimap.nii.gz'
    )
    # In a template space requested without a resolution
    assert _sink_scalar(tmp_path, source_name, space='MNI152NLin2009cAsym') == (
        'sub-01_ses-1_space-MNI152NLin2009cAsym_model-dki_param-fa_dwimap.nii.gz'
    )
    # In a template space requested with one: the requested resolution wins
    assert _sink_scalar(tmp_path, source_name, space='MNI152NLin2009cAsym', res='2') == (
        'sub-01_ses-1_space-MNI152NLin2009cAsym_res-2_model-dki_param-fa_dwimap.nii.gz'
    )

    # The tables named with get_recon_output_name follow the same rule
    bids_dir = tmp_path / 'layout'
    bids_dir.mkdir()
    layout = BIDSLayout(bids_dir, validate=False, config=['bids', 'derivatives'])
    monkeypatch.setattr(config.execution, 'layout', layout)
    out_name = get_recon_output_name(
        base_dir=str(tmp_path / 'out'),
        source_file=str(tmp_path / 'in' / 'sub-01' / 'ses-1' / 'dwi' / source_name),
        derivative_file=str(tmp_path / 'stats.tsv'),
        output_bids_entities={'suffix': 'scalarstats', 'seg': 'AAL116'},
        dismiss_entities=['desc'],
    )
    assert out_name.replace('\\', '/').endswith(
        '/sub-01/ses-1/dwi/sub-01_ses-1_space-ACPC_seg-AAL116_scalarstats.tsv'
    )
