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
    """Repeats collapse, and several spaces and resolutions are kept in the order given."""
    specs = spaces.parse_output_spaces(['MNIInfant:cohort-2', 'MNIInfant:cohort-2'])
    assert specs == [SpaceSpec(space='MNIInfant', cohort='2')]

    specs = spaces.parse_output_spaces(
        ['MNI152NLin2009cAsym:res-1:res-2mm', 'MNI152NLin6Asym', 'MNI152NLin2009cAsym:res-2mm'],
    )
    assert [str(spec) for spec in specs] == [
        'MNI152NLin2009cAsym:res-1',
        'MNI152NLin2009cAsym:res-2mm',
        'MNI152NLin6Asym',
    ]


def test_space_spec_properties():
    """SpaceSpec reports its name, TemplateFlow resolution and transform matches."""
    (adult,) = spaces.parse_space_token('MNI152NLin2009cAsym:res-2')
    assert adult.fullname == 'MNI152NLin2009cAsym'
    assert adult.templateflow_resolution == '2'
    assert adult.res_label == '2'
    assert adult.node_label == 'MNI152NLin2009cAsym_res_2'
    assert not adult.is_infant
    assert adult.matches('MNI152NLin2009cAsym')
    assert not adult.matches('MNI152NLin6Asym')
    assert not adult.matches('MNI152NLin2009cAsym+1')

    (infant,) = spaces.parse_space_token('MNIInfant:cohort-2:res-2mm')
    assert infant.fullname == 'MNIInfant+2'
    assert infant.templateflow_resolution == '1'
    assert infant.res_label == '2mm'
    assert infant.node_label == 'MNIInfant_cohort_2_res_2mm'
    assert infant.is_infant
    assert infant.matches('MNIInfant+2')
    assert not infant.matches('MNIInfant+1')
    assert not infant.matches('MNIInfant')

    (any_cohort,) = spaces.parse_space_token('MNIInfant:cohort-auto')
    assert any_cohort.fullname == 'MNIInfant'
    # No resolution was requested, so outputs in this space have no res- entity
    assert any_cohort.res_label is None
    assert any_cohort.matches('MNIInfant+1')
    assert any_cohort.matches('MNIInfant+2')
    assert not any_cohort.matches('MNI152NLin2009cAsym')


@pytest.mark.parametrize(
    'token',
    [
        'MNI152NLin2009cAsym',
        'MNI152NLin2009cAsym:res-2mm',
        'MNI152NLin2009cAsym:res-1p5x1p5x3mm',
        'MNI152NLin6Asym:res-2',
        'MNIInfant:cohort-2:res-1p5mm',
        'MNIInfant:cohort-auto',
    ],
)
def test_space_spec_from_token(token, monkeypatch):
    """A canonical token is rebuilt into the same spec without asking TemplateFlow."""
    (expected,) = spaces.parse_space_token(token)

    def _fail(*args):
        raise AssertionError('TemplateFlow was consulted')

    for name in ('_templates', '_cohorts', '_resolutions'):
        monkeypatch.setattr(spaces, name, _fail)

    assert SpaceSpec.from_token(token) == expected


def test_resolve_auto_cohort():
    """cohort-auto becomes the cohort the dataset has a transform to."""
    names = ['MNI152NLin2009cAsym', 'MNIInfant+2']
    (auto,) = spaces.parse_space_token('MNIInfant:cohort-auto:res-2mm')
    (named,) = spaces.parse_space_token('MNIInfant:cohort-2:res-2mm')
    (adult,) = spaces.parse_space_token('MNI152NLin2009cAsym')

    assert spaces.resolve_auto_cohort(auto, names) == named
    # Specs that name their cohort, or have none, are left alone
    assert spaces.resolve_auto_cohort(named, names) is named
    assert spaces.resolve_auto_cohort(adult, names) is adult
    (other,) = spaces.parse_space_token('MNIInfant:cohort-1')
    assert spaces.resolve_auto_cohort(other, names) is other

    with pytest.raises(OutputSpacesError, match='has no transform to MNIInfant'):
        spaces.resolve_auto_cohort(auto, ['MNI152NLin2009cAsym'])

    with pytest.raises(OutputSpacesError, match='Name one cohort in --output-spaces'):
        spaces.resolve_auto_cohort(auto, [*names, 'MNIInfant+3'])


def test_zoom_formatting():
    """Voxel sizes are spelled for filenames and for people."""
    assert spaces.zooms_label((2.0, 2.0, 2.0)) == '2mm'
    assert spaces.zooms_label((1.5, 1.5, 3.0)) == '1p5x1p5x3mm'
    assert spaces.format_zooms((2.0, 2.0, 2.0)) == '2 mm isotropic'
    assert spaces.format_zooms((1.5, 1.5, 3.0)) == '1.5 x 1.5 x 3 mm'


def test_config_output_space_specs(output_spaces_setting):
    """The config returns the default space until others are requested."""
    config.workflow.output_spaces = None
    assert config.workflow.output_space_specs() == [SpaceSpec(space='MNI152NLin2009cAsym')]

    config.workflow.output_spaces = ['MNIInfant:cohort-2:res-2mm', 'MNI152NLin6Asym:res-2']
    infant, adult = config.workflow.output_space_specs()
    assert infant.fullname == 'MNIInfant+2'
    assert infant.resolution.zooms == (2.0, 2.0, 2.0)
    assert adult.fullname == 'MNI152NLin6Asym'
    assert adult.templateflow_resolution == '2'


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

    # Several spaces, and several resolutions of one space, are all kept
    opts = _parse(
        [*cli_args, '--output-spaces', 'MNI152NLin2009cAsym:res-1:res-2mm', 'MNI152NLin6Asym'],
    )
    assert opts.output_spaces == [
        'MNI152NLin2009cAsym:res-1',
        'MNI152NLin2009cAsym:res-2mm',
        'MNI152NLin6Asym',
    ]


@pytest.mark.parametrize(
    ('extra', 'message'),
    [
        (['--output-spaces', 'acpc'], 'is not an output space in QSIRecon'),
        (['--output-spaces', 'MNI152NLin2009cAsym', 'NotATemplate'], 'is not a known output'),
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


def _collect(bids_dir, *tokens, needs_anat_transform=False):
    layout = BIDSLayout(bids_dir, validate=False, config=['bids', 'derivatives'])
    return xbids.collect_anatomical_data(
        layout=layout,
        subject_id='01',
        session_id=Query.NONE,
        needs_anat_transform=needs_anat_transform,
        output_spaces=spaces.parse_output_spaces(tokens),
        bids_filters={},
    )


def _names(files):
    return [str(f).replace('\\', '/').rsplit('/', 1)[-1] for f in files]


@pytest.mark.parametrize(
    ('token', 'resolved', 'template'),
    [
        ('MNI152NLin2009cAsym', 'MNI152NLin2009cAsym', 'MNI152NLin2009cAsym'),
        ('MNI152NLin2009cAsym:res-2mm', 'MNI152NLin2009cAsym:res-2mm', 'MNI152NLin2009cAsym'),
        ('MNIInfant:cohort-2', 'MNIInfant:cohort-2', 'MNIInfant+2'),
        ('MNIInfant:cohort-auto', 'MNIInfant:cohort-2', 'MNIInfant+2'),
    ],
)
def test_collect_anatomical_data_output_space(infant_layout, token, resolved, template):
    """The transform to the requested template (and cohort) is the one collected."""
    anat_data, status = _collect(infant_layout, token)
    assert status['has_qsiprep_anat_transforms'] is True
    # The template transforms are not returned as workflow inputs
    assert 'acpc_to_template_xfm' not in anat_data
    assert 'template_to_acpc_xfm' not in anat_data

    (space_xfms,) = status['output_space_xfms']
    assert space_xfms['token'] == resolved
    assert space_xfms['space'] == template
    assert space_xfms['transform_path'] is None
    assert _names(space_xfms['acpc_to_template_xfm']) == [
        f'sub-01_from-ACPC_to-{template}_mode-image_xfm.h5',
    ]


def test_collect_anatomical_data_several_spaces(infant_layout):
    """Every requested space and resolution gets its transforms, in the order requested."""
    import json

    _, status = _collect(
        infant_layout,
        'MNI152NLin2009cAsym:res-1:res-2mm',
        'MNIInfant:cohort-2',
        'MNI152NLin2009cAsym',
    )
    resolved = status['output_space_xfms']
    assert [space_xfms['token'] for space_xfms in resolved] == [
        'MNI152NLin2009cAsym:res-1',
        'MNI152NLin2009cAsym:res-2mm',
        'MNIInfant:cohort-2',
        'MNI152NLin2009cAsym',
    ]
    assert [space_xfms['space'] for space_xfms in resolved] == [
        'MNI152NLin2009cAsym',
        'MNI152NLin2009cAsym',
        'MNIInfant+2',
        'MNI152NLin2009cAsym',
    ]
    # Resolutions of one template share its transforms
    adult = resolved[0]['acpc_to_template_xfm']
    assert _names(adult) == ['sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5']
    assert resolved[1]['acpc_to_template_xfm'] == adult
    assert resolved[3]['acpc_to_template_xfm'] == adult
    assert _names(resolved[2]['acpc_to_template_xfm']) == [
        'sub-01_from-ACPC_to-MNIInfant+2_mode-image_xfm.h5',
    ]
    # The entries travel in the dictionary given to every reconstruction workflow,
    # which may be written to a sidecar
    assert json.loads(json.dumps(resolved)) == resolved


def test_collect_anatomical_data_auto_cohort_is_deduplicated(infant_layout):
    """cohort-auto is resolved before repeats are dropped, so it cannot repeat a space."""
    _, status = _collect(infant_layout, 'MNIInfant:cohort-2', 'MNIInfant:cohort-auto')
    assert [space_xfms['token'] for space_xfms in status['output_space_xfms']] == [
        'MNIInfant:cohort-2',
    ]

    # Different resolutions of the resolved space are still different outputs
    _, status = _collect(
        infant_layout,
        'MNIInfant:cohort-auto:res-2mm',
        'MNIInfant:cohort-2:res-2mm',
        'MNIInfant:cohort-auto',
    )
    assert [space_xfms['token'] for space_xfms in status['output_space_xfms']] == [
        'MNIInfant:cohort-2:res-2mm',
        'MNIInfant:cohort-2',
    ]


@pytest.mark.parametrize('needs_anat_transform', [False, True])
@pytest.mark.parametrize('token', ['MNIInfant:cohort-1', 'MNI152NLin6Asym'])
def test_collect_anatomical_data_missing_space(infant_layout, token, needs_anat_transform):
    """Requesting a space the dataset was not normalized to names the ones it has."""
    with pytest.raises(ValueError, match='was requested') as excinfo:
        _collect(infant_layout, token, needs_anat_transform=needs_anat_transform)

    message = str(excinfo.value)
    assert f'The output space {token} was requested' in message
    assert 'It has transforms to: MNI152NLin2009cAsym, MNIInfant:cohort-2.' in message
    assert 'TemplateFlow has no transforms that link them' in message


def test_collect_anatomical_data_one_unreachable_space_is_an_error(infant_layout):
    """A space that cannot be reached stops the build even when the others can."""
    with pytest.raises(ValueError, match='The output space MNI152NLin6Asym was requested'):
        _collect(infant_layout, 'MNI152NLin2009cAsym', 'MNI152NLin6Asym')


def test_collect_anatomical_data_one_way_space(infant_layout):
    """Scalar maps only go into the template, so only that direction is needed."""
    anat_dir = infant_layout / 'sub-01' / 'anat'
    (anat_dir / 'sub-01_from-MNIInfant+2_to-ACPC_mode-image_xfm.h5').unlink()

    _, status = _collect(infant_layout, 'MNIInfant:cohort-2')
    (space_xfms,) = status['output_space_xfms']
    assert _names(space_xfms['acpc_to_template_xfm']) == [
        'sub-01_from-ACPC_to-MNIInfant+2_mode-image_xfm.h5',
    ]

    # The transform out of the template is of no use for an output space
    (anat_dir / 'sub-01_from-MNIInfant+2_to-ACPC_mode-image_xfm.h5').touch()
    (anat_dir / 'sub-01_from-ACPC_to-MNIInfant+2_mode-image_xfm.h5').unlink()
    with pytest.raises(ValueError, match=r'It has transforms to: MNI152NLin2009cAsym\. '):
        _collect(infant_layout, 'MNIInfant:cohort-2')


def test_collect_anatomical_data_no_transforms(infant_layout):
    """A dataset with no template transforms is reported, and fails only if they are needed."""
    for xfm in (infant_layout / 'sub-01' / 'anat').glob('*_xfm.h5'):
        xfm.unlink()

    anat_data, status = _collect(infant_layout, 'MNI152NLin2009cAsym', 'MNI152NLin6Asym')
    assert status['has_qsiprep_anat_transforms'] is False
    assert status['output_space_xfms'] == []
    assert status['odf_rois_to_acpc_xfm'] is None
    assert 'acpc_to_template_xfm' not in anat_data

    with pytest.raises(ValueError, match='no "acpc_to_template_xfm" file was found'):
        _collect(infant_layout, 'MNI152NLin2009cAsym', needs_anat_transform=True)


def test_collect_anatomical_data_ambiguous_cohort(infant_layout):
    """cohort-auto cannot choose between the transforms of two cohorts."""
    anat_dir = infant_layout / 'sub-01' / 'anat'
    for xfm in list(anat_dir.glob('*MNIInfant+2*')):
        shutil.copy(xfm, anat_dir / xfm.name.replace('MNIInfant+2', 'MNIInfant+3'))

    with pytest.raises(ValueError, match='Name one cohort in --output-spaces'):
        _collect(infant_layout, 'MNIInfant:cohort-auto')

    # Naming the cohorts gives both
    _, status = _collect(infant_layout, 'MNIInfant:cohort-3', 'MNIInfant:cohort-2')
    assert [space_xfms['space'] for space_xfms in status['output_space_xfms']] == [
        'MNIInfant+3',
        'MNIInfant+2',
    ]


def test_collect_anatomical_data_duplicate_transforms(infant_layout):
    """Two transforms to one template cannot be chosen between."""
    anat_dir = infant_layout / 'sub-01' / 'anat'
    name = 'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    shutil.copy(anat_dir / name, anat_dir / name.replace('sub-01_', 'sub-01_acq-b_'))

    with pytest.raises(ValueError, match='More than one from-ACPC_to-MNI152NLin2009cAsym') as exc:
        _collect(infant_layout, 'MNI152NLin2009cAsym')

    assert '--bids-filter-file' in str(exc.value)

    # A duplicate that is not on the way to a requested space does no harm
    _, status = _collect(infant_layout, 'MNIInfant:cohort-2')
    assert len(status['output_space_xfms']) == 1


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
    wf = init_output_grid_wf(output_space=output_space, name='grid_wf')

    assert wf.name == 'grid_wf'
    optional = {'voxel_size_chooser', 'resample_to_voxel_size'}
    assert optional.intersection(wf.list_node_names()) == nodes
    assert wf.get_node('autobox_template').inputs.padding == padding
    if 'resample_to_voxel_size' in nodes:
        fixed = wf.get_node('resample_to_voxel_size').inputs.voxel_size
        if voxel_size is None:
            assert not isdefined(fixed)
        else:
            assert fixed == voxel_size


@pytest.mark.parametrize('has_transforms', [False, True])
def test_dwi_anatomical_wf_passes_output_spaces_on(tmp_path, monkeypatch, has_transforms):
    """The DWI anatomical workflow builds no template grid; it hands the transforms on."""
    from qsirecon.workflows.recon.anatomical import init_dwi_recon_anatomical_workflow

    monkeypatch.setattr(config.execution, 'output_dir', tmp_path)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)

    output_space_xfms = []
    if has_transforms:
        output_space_xfms = [
            {
                'token': 'MNIInfant:cohort-2:res-2mm',
                'space': 'MNIInfant+2',
                'acpc_to_template_xfm': ['from-ACPC_to-MNIInfant+2_xfm.h5'],
                'transform_path': None,
            },
        ]

    wf, status = init_dwi_recon_anatomical_workflow(
        atlas_configs={},
        has_qsiprep_5tt_hsvs=False,
        needs_anat_transform=False,
        has_freesurfer_5tt_hsvs=False,
        has_qsiprep_anat=True,
        has_qsiprep_anat_transforms=has_transforms,
        has_freesurfer=False,
        extras_to_make=[],
        name='dwi_specific_anat_wf',
        output_space_xfms=output_space_xfms,
    )

    assert status['output_space_xfms'] == output_space_xfms
    assert 'template_output_space' not in status
    nodes = wf.list_node_names()
    assert not any('output_grid' in name or 'get_template' in name for name in nodes)
    assert 'resampling_template' not in wf.get_node('outputnode').inputs.copyable_trait_names()


def test_subject_summary_names_every_space(output_spaces_setting):
    """The report summary lists every requested space and resolution."""
    from qsirecon.workflows.base import _summarize_output_spaces

    statuses = [
        {
            'output_space_xfms': [
                {'token': 'MNI152NLin2009cAsym:res-1'},
                {'token': 'MNI152NLin2009cAsym:res-2mm'},
                {'token': 'MNIInfant:cohort-2'},
            ],
        },
        # A second anatomical image of the same subject resolves the same spaces
        {'output_space_xfms': [{'token': 'MNIInfant:cohort-2'}]},
        {'output_space_xfms': []},
    ]
    assert _summarize_output_spaces(statuses) == [
        'MNI152NLin2009cAsym (res-1)',
        'MNI152NLin2009cAsym (res-2mm)',
        'MNIInfant+2',
    ]

    # With nothing resolved, the spaces are named as they were requested
    config.workflow.output_spaces = None
    assert _summarize_output_spaces([{'output_space_xfms': []}]) == ['MNI152NLin2009cAsym']
    config.workflow.output_spaces = ['MNI152NLin6Asym:res-2', 'MNIInfant:cohort-auto']
    assert _summarize_output_spaces([]) == ['MNI152NLin6Asym (res-2)', 'MNIInfant']
