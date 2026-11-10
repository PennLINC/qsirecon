"""Tests for --output-spaces parsing and the code that consumes it."""

import shutil

import pytest
from bids.layout import BIDSLayout, Query
from niworkflows.utils.testing import generate_bids_skeleton

import qsirecon.utils.bids as xbids
from qsirecon import config
from qsirecon.cli import parser as cli_parser
from qsirecon.cli import version as _version
from qsirecon.data import load as load_data
from qsirecon.utils import spaces, transforms
from qsirecon.utils.spaces import OutputSpacesError, Resolution, SpaceSpec

TEMPLATES = {
    'MNI152NLin2009cAsym': {'cohorts': [], 'resolutions': ['1', '2']},
    'MNI152NLin6Asym': {'cohorts': [], 'resolutions': ['1', '2']},
    'MNIInfant': {'cohorts': ['1', '2', '3'], 'resolutions': ['1', '2']},
}


@pytest.fixture(autouse=True)
def fake_templateflow(monkeypatch):
    """Answer TemplateFlow queries without a TemplateFlow skeleton or network access."""
    monkeypatch.setattr(spaces, '_templates', lambda: list(TEMPLATES))
    monkeypatch.setattr(spaces, '_cohorts', lambda template: TEMPLATES[template]['cohorts'])
    monkeypatch.setattr(
        spaces,
        '_resolutions',
        lambda template: TEMPLATES[template]['resolutions'],
    )


@pytest.fixture(autouse=True)
def no_templateflow_transforms(monkeypatch):
    """Leave TemplateFlow without template-to-template transforms unless a test adds them."""
    monkeypatch.setattr(transforms, '_tf_templates', list)


@pytest.fixture
def output_spaces_setting():
    """Restore config.workflow.output_spaces after a test changes it."""
    original = config.workflow.output_spaces
    yield
    config.workflow.output_spaces = original


@pytest.mark.parametrize(
    ('token', 'expected', 'canonical'),
    [
        (
            'MNI152NLin2009cAsym',
            SpaceSpec(space='MNI152NLin2009cAsym'),
            'MNI152NLin2009cAsym',
        ),
        (
            'MNI152NLin2009cAsym:res-2mm',
            SpaceSpec(
                space='MNI152NLin2009cAsym',
                resolution=Resolution(kind='mm', label='2mm', zooms=(2.0, 2.0, 2.0)),
            ),
            'MNI152NLin2009cAsym:res-2mm',
        ),
        (
            'MNI152NLin2009cAsym:res-1p5x1p5x3mm',
            SpaceSpec(
                space='MNI152NLin2009cAsym',
                resolution=Resolution(kind='mm', label='1p5x1p5x3mm', zooms=(1.5, 1.5, 3.0)),
            ),
            'MNI152NLin2009cAsym:res-1p5x1p5x3mm',
        ),
        (
            'MNI152NLin6Asym:res-2',
            SpaceSpec(
                space='MNI152NLin6Asym',
                resolution=Resolution(kind='label', label='2'),
            ),
            'MNI152NLin6Asym:res-2',
        ),
        (
            'MNIInfant:res-1p5mm:cohort-2',
            SpaceSpec(
                space='MNIInfant',
                resolution=Resolution(kind='mm', label='1p5mm', zooms=(1.5, 1.5, 1.5)),
                cohort='2',
            ),
            'MNIInfant:cohort-2:res-1p5mm',
        ),
        (
            'MNIInfant:cohort-auto',
            SpaceSpec(space='MNIInfant', cohort='auto'),
            'MNIInfant:cohort-auto',
        ),
    ],
)
def test_parse_space_token(token, expected, canonical):
    """Tokens parse to specs, and specs print back as canonical tokens."""
    (spec,) = spaces.parse_space_token(token)
    assert spec == expected
    assert str(spec) == canonical
    # The canonical form parses back to the same spec
    assert spaces.parse_space_token(canonical) == [expected]


def test_parse_space_token_expands_resolutions():
    """Several res- specs on one token give one spec each."""
    specs = spaces.parse_space_token('MNI152NLin2009cAsym:res-1:res-3mm')
    assert [str(spec) for spec in specs] == [
        'MNI152NLin2009cAsym:res-1',
        'MNI152NLin2009cAsym:res-3mm',
    ]


@pytest.mark.parametrize(
    ('token', 'message'),
    [
        ('acpc:res-2mm', 'is not an output space in QSIRecon'),
        ('anat', 'is not an output space in QSIRecon'),
        ('NotATemplate', 'is not a known output space'),
        ('MNI152NLin2009cAsym:res', 'is not a key-value specification'),
        ('MNI152NLin2009cAsym:den-91k', 'is not a valid specifier'),
        ('MNI152NLin2009cAsym:res-3', 'has no res-3'),
        ('MNI152NLin2009cAsym:res-nativemin', 'is not valid in QSIRecon'),
        ('MNI152NLin2009cAsym:res-2x2mm', 'needs one or three sizes'),
        ('MNI152NLin2009cAsym:res-0mm', 'must be greater than zero'),
        ('MNI152NLin2009cAsym:res-2pmm', 'has no res-2pmm'),
        ('MNI152NLin2009cAsym:res-p5mm', 'has no res-p5mm'),
        ('MNI152NLin2009cAsym:cohort-1', 'does not accept a cohort'),
        ('MNIInfant', 'needs a cohort'),
        ('MNIInfant:cohort-9', 'has no cohort-9'),
        ('MNIInfant:cohort-1:cohort-2', 'more than one cohort'),
    ],
)
def test_parse_space_token_errors(token, message):
    """Invalid tokens are rejected with a message that says why."""
    with pytest.raises(OutputSpacesError, match=message):
        spaces.parse_space_token(token)


def test_parse_output_spaces():
    """Repeats collapse, and more than one distinct space is rejected."""
    specs = spaces.parse_output_spaces(['MNIInfant:cohort-2', 'MNIInfant:cohort-2'])
    assert specs == [SpaceSpec(space='MNIInfant', cohort='2')]

    with pytest.raises(OutputSpacesError, match='one space at one resolution'):
        spaces.parse_output_spaces(['MNI152NLin2009cAsym', 'MNI152NLin6Asym'])

    with pytest.raises(OutputSpacesError, match='one space at one resolution'):
        spaces.parse_output_spaces(['MNI152NLin2009cAsym:res-1mm:res-2mm'])


def test_space_spec_properties():
    """SpaceSpec reports its name, TemplateFlow resolution and transform matches."""
    (adult,) = spaces.parse_space_token('MNI152NLin2009cAsym:res-2')
    assert adult.fullname == 'MNI152NLin2009cAsym'
    assert adult.templateflow_resolution == '2'
    assert not adult.is_infant
    assert adult.matches('MNI152NLin2009cAsym')
    assert not adult.matches('MNI152NLin6Asym')
    assert not adult.matches('MNI152NLin2009cAsym+1')

    (infant,) = spaces.parse_space_token('MNIInfant:cohort-2:res-2mm')
    assert infant.fullname == 'MNIInfant+2'
    assert infant.templateflow_resolution == '1'
    assert infant.is_infant
    assert infant.matches('MNIInfant+2')
    assert not infant.matches('MNIInfant+1')
    assert not infant.matches('MNIInfant')

    (any_cohort,) = spaces.parse_space_token('MNIInfant:cohort-auto')
    assert any_cohort.fullname == 'MNIInfant'
    assert any_cohort.matches('MNIInfant+1')
    assert any_cohort.matches('MNIInfant+2')
    assert not any_cohort.matches('MNI152NLin2009cAsym')


def test_zoom_formatting():
    """Voxel sizes are spelled for filenames and for people."""
    assert spaces.zooms_label((2.0, 2.0, 2.0)) == '2mm'
    assert spaces.zooms_label((1.5, 1.5, 3.0)) == '1p5x1p5x3mm'
    assert spaces.format_zooms((2.0, 2.0, 2.0)) == '2 mm isotropic'
    assert spaces.format_zooms((1.5, 1.5, 3.0)) == '1.5 x 1.5 x 3 mm'


def test_config_output_space(output_spaces_setting):
    """The config returns the default space until one is requested."""
    config.workflow.output_spaces = None
    assert config.workflow.output_space() == SpaceSpec(space='MNI152NLin2009cAsym')

    config.workflow.output_spaces = ['MNIInfant:cohort-2:res-2mm']
    spec = config.workflow.output_space()
    assert spec.fullname == 'MNIInfant+2'
    assert spec.resolution.zooms == (2.0, 2.0, 2.0)


@pytest.fixture
def cli_args(tmp_path, monkeypatch):
    """Arguments for a minimal command line, without the version check."""
    monkeypatch.setattr(_version, 'check_latest', lambda *args, **kwargs: None)
    monkeypatch.setattr(_version, 'is_flagged', lambda *args, **kwargs: (False, None))
    input_dir = tmp_path / 'qsiprep'
    input_dir.mkdir()
    return [str(input_dir), str(tmp_path / 'out'), 'participant', '--recon-spec', 'dipy_dki']


def _parse(args):
    parser = cli_parser._build_parser()
    opts = parser.parse_args(args)
    return cli_parser._finalize_output_spaces(opts, parser)


def test_cli_output_spaces(cli_args):
    """--output-spaces is stored as canonical tokens, and left unset by default."""
    assert _parse(cli_args).output_spaces is None

    opts = _parse([*cli_args, '--output-spaces', 'MNIInfant:res-2mm:cohort-2'])
    assert opts.output_spaces == ['MNIInfant:cohort-2:res-2mm']

    # The same space given twice is one request
    opts = _parse(
        [
            *cli_args,
            '--output-spaces',
            'MNI152NLin2009cAsym:res-2',
            '--output-spaces',
            'MNI152NLin2009cAsym:res-2',
        ]
    )
    assert opts.output_spaces == ['MNI152NLin2009cAsym:res-2']


@pytest.mark.parametrize(
    ('extra', 'message'),
    [
        (['--output-spaces', 'acpc'], 'is not an output space in QSIRecon'),
        (['--output-spaces', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'], 'one space at one'),
        (['--output-spaces', 'MNI152NLin2009cAsym:res-1:res-2'], 'one space at one'),
        (['--output-resolution', '2'], 'MNI152NLin2009cAsym:res-2mm'),
        (['--output-resolution', '1.7'], 'MNI152NLin2009cAsym:res-1p7mm'),
        (['--infant'], 'MNIInfant:cohort-auto'),
    ],
)
def test_cli_output_spaces_errors(cli_args, capsys, extra, message):
    """Bad requests and the removed flags stop the parser with a helpful message."""
    with pytest.raises(SystemExit):
        _parse([*cli_args, *extra])

    assert message in capsys.readouterr().err


@pytest.fixture
def infant_layout(tmp_path):
    """A QSIPrep-like dataset with adult and infant (cohort 2) template transforms."""
    skeleton = load_data('tests/skeletons/longitudinal_anat.yml')
    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    anat_dir = bids_dir / 'sub-01' / 'anat'
    for direction in ('from-ACPC_to-{}', 'from-{}_to-ACPC'):
        adult = anat_dir / f'sub-01_{direction.format("MNI152NLin2009cAsym")}_mode-image_xfm.h5'
        assert adult.is_file()
        shutil.copy(
            adult,
            anat_dir / f'sub-01_{direction.format("MNIInfant+2")}_mode-image_xfm.h5',
        )

    return bids_dir


def _collect(bids_dir, token, needs_t1w_transform=False):
    layout = BIDSLayout(bids_dir, validate=False, config=['bids', 'derivatives'])
    (output_space,) = spaces.parse_output_spaces([token])
    return xbids.collect_anatomical_data(
        layout=layout,
        subject_id='01',
        session_id=Query.NONE,
        needs_t1w_transform=needs_t1w_transform,
        output_space=output_space,
        bids_filters={},
    )


@pytest.mark.parametrize(
    ('token', 'template'),
    [
        ('MNI152NLin2009cAsym', 'MNI152NLin2009cAsym'),
        ('MNI152NLin2009cAsym:res-2mm', 'MNI152NLin2009cAsym'),
        ('MNIInfant:cohort-2', 'MNIInfant+2'),
        ('MNIInfant:cohort-auto', 'MNIInfant+2'),
    ],
)
def test_collect_anatomical_data_output_space(infant_layout, token, template):
    """The transforms of the requested template (and cohort) are the ones collected."""
    anat_data, status = _collect(infant_layout, token)
    assert status['has_qsiprep_t1w_transforms'] is True
    assert status['template_output_space'] == template
    assert f'from-ACPC_to-{template}_' in anat_data['acpc_to_template_xfm']
    assert f'from-{template}_to-ACPC_' in anat_data['template_to_acpc_xfm']


@pytest.mark.parametrize('needs_t1w_transform', [False, True])
@pytest.mark.parametrize('token', ['MNIInfant:cohort-1', 'MNI152NLin6Asym'])
def test_collect_anatomical_data_missing_space(infant_layout, token, needs_t1w_transform):
    """Requesting a space the dataset was not normalized to names the ones it has."""
    with pytest.raises(ValueError, match='was requested') as excinfo:
        _collect(infant_layout, token, needs_t1w_transform=needs_t1w_transform)

    message = str(excinfo.value)
    assert f'The output space {token} was requested' in message
    assert 'It has transforms for: MNI152NLin2009cAsym, MNIInfant:cohort-2.' in message
    assert 'TemplateFlow has no transforms that link them' in message


def test_collect_anatomical_data_incomplete_space(infant_layout):
    """One direction of the requested space's transforms is not enough."""
    anat_dir = infant_layout / 'sub-01' / 'anat'
    (anat_dir / 'sub-01_from-MNIInfant+2_to-ACPC_mode-image_xfm.h5').unlink()

    with pytest.raises(ValueError, match=r'It has transforms for: MNI152NLin2009cAsym\. '):
        _collect(infant_layout, 'MNIInfant:cohort-2')


def test_collect_anatomical_data_no_transforms(infant_layout):
    """A dataset with no template transforms is reported, and fails only if they are needed."""
    for xfm in (infant_layout / 'sub-01' / 'anat').glob('*_xfm.h5'):
        xfm.unlink()

    anat_data, status = _collect(infant_layout, 'MNI152NLin2009cAsym')
    assert status['has_qsiprep_t1w_transforms'] is False
    assert status['template_output_space'] is None
    assert anat_data['acpc_to_template_xfm'] is None
    assert anat_data['template_to_acpc_xfm'] is None

    with pytest.raises(ValueError, match='requested output space'):
        _collect(infant_layout, 'MNI152NLin2009cAsym', needs_t1w_transform=True)


def test_collect_anatomical_data_ambiguous_cohort(infant_layout):
    """cohort-auto cannot choose between the transforms of two cohorts."""
    anat_dir = infant_layout / 'sub-01' / 'anat'
    for xfm in list(anat_dir.glob('*MNIInfant+2*')):
        shutil.copy(xfm, anat_dir / xfm.name.replace('MNIInfant+2', 'MNIInfant+3'))

    with pytest.raises(ValueError, match='Name one cohort in --output-spaces'):
        _collect(infant_layout, 'MNIInfant:cohort-auto')

    _, status = _collect(infant_layout, 'MNIInfant:cohort-3')
    assert status['template_output_space'] == 'MNIInfant+3'


@pytest.mark.parametrize(
    ('token', 'nodes', 'voxel_size', 'padding'),
    [
        # No resolution: the voxel size is read from the DWI at run time
        (
            'MNI152NLin2009cAsym',
            {'voxel_size_chooser', 'resample_to_voxel_size'},
            None,
            8,
        ),
        ('MNI152NLin2009cAsym:res-2mm', {'resample_to_voxel_size'}, (2.0, 2.0, 2.0), 8),
        ('MNIInfant:cohort-2:res-1p5x1p5x3mm', {'resample_to_voxel_size'}, (1.5, 1.5, 3.0), 4),
        # A TemplateFlow label: the fetched template's voxel size, with no resampling
        ('MNI152NLin2009cAsym:res-2', set(), None, 8),
    ],
)
def test_init_output_grid_wf(token, nodes, voxel_size, padding):
    """The grid workflow resamples to the requested voxel size, when there is one."""
    from nipype.interfaces.base import isdefined

    from qsirecon.workflows.recon.anatomical import init_output_grid_wf

    (output_space,) = spaces.parse_output_spaces([token])
    wf = init_output_grid_wf(output_space=output_space)

    optional = {'voxel_size_chooser', 'resample_to_voxel_size'}
    assert optional.intersection(wf.list_node_names()) == nodes
    assert wf.get_node('autobox_template').inputs.padding == padding
    if 'resample_to_voxel_size' in nodes:
        fixed = wf.get_node('resample_to_voxel_size').inputs.voxel_size
        if voxel_size is None:
            assert not isdefined(fixed)
        else:
            assert fixed == voxel_size


@pytest.mark.parametrize('template', [None, 'MNI152NLin2009cAsym', 'MNIInfant+2'])
def test_dwi_anatomical_wf_template_grid(tmp_path, monkeypatch, template):
    """The template grid is only built when there are transforms to a template."""
    from qsirecon.workflows.recon.anatomical import init_dwi_recon_anatomical_workflow

    monkeypatch.setattr(config.execution, 'output_dir', tmp_path)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)

    wf, status = init_dwi_recon_anatomical_workflow(
        atlas_configs={},
        has_qsiprep_5tt_hsvs=False,
        needs_t1w_transform=False,
        has_freesurfer_5tt_hsvs=False,
        has_qsiprep_t1w=True,
        has_qsiprep_t1w_transforms=template is not None,
        has_freesurfer=False,
        extras_to_make=[],
        name='dwi_specific_anat_wf',
        template_output_space=template,
    )

    assert status['template_output_space'] == template
    nodes = wf.list_node_names()
    has_grid = any(name.startswith('output_grid_wf.') for name in nodes)
    assert has_grid is (template is not None)
    assert ('get_template' in nodes) is (template is not None)
    if template:
        assert wf.get_node('get_template').inputs.template_name == template


def test_scalar_to_template_wf_needs_template():
    """Mapping scalars to a template fails clearly when there is no template space."""
    from qsirecon.workflows.recon.scalar_mapping import init_scalar_to_template_wf

    with pytest.raises(ValueError, match='cannot be warped to a template'):
        init_scalar_to_template_wf(inputs_dict={'template_output_space': None})
