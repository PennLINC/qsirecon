"""Tests for chaining the input dataset's transforms with TemplateFlow's."""

import shutil

import pytest
from bids.layout import BIDSLayout, Query
from niworkflows.utils.testing import generate_bids_skeleton

import qsirecon.utils.bids as xbids
from qsirecon.data import load as load_data
from qsirecon.utils import spaces, transforms
from qsirecon.utils.transforms import NoTransformChainError, Xfm

# What TemplateFlow lists: the adult templates are linked to each other, and the infant
# cohorts are linked to MNI152NLin6Asym only.
TF_XFMS = {
    'MNI152NLin2009cAsym': ['tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5'],
    'MNI152NLin6Asym': [
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNIInfant+2_mode-image_xfm.h5',
        # One direction only, so it cannot be used
        'tpl-MNI152NLin6Asym_from-OASIS30ANTs_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-points_xfm.h5',
    ],
    'MNIInfant': ['tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5'],
    'OASIS30ANTs': [],
}
TEMPLATES = {
    'MNI152NLin2009cAsym': [],
    'MNI152NLin6Asym': [],
    'MNIInfant': ['1', '2', '3'],
    'OASIS30ANTs': [],
}


@pytest.fixture(autouse=True)
def fake_templateflow(monkeypatch, tmp_path):
    """Answer TemplateFlow queries from TF_XFMS, and record what is downloaded."""
    tf_home = tmp_path / 'templateflow'
    fetched = []

    def _ls(template):
        return [str(tf_home / f'tpl-{template}' / name) for name in TF_XFMS[template]]

    def _get(template, **query):
        cohort = f'_cohort-{query["cohort"]}' if 'cohort' in query else ''
        mode = f'_mode-{query["mode"]}' if 'mode' in query else ''
        name = f'tpl-{template}{cohort}_from-{query["from"]}{mode}_xfm.h5'
        if name not in TF_XFMS[template]:
            return []

        fetched.append(name)
        return tf_home / f'tpl-{template}' / name

    monkeypatch.setattr(transforms, '_tf_templates', lambda: list(TF_XFMS))
    monkeypatch.setattr(transforms, '_tf_ls', _ls)
    monkeypatch.setattr(transforms, '_tf_get', _get)
    monkeypatch.setattr(spaces, '_templates', lambda: list(TEMPLATES))
    monkeypatch.setattr(spaces, '_cohorts', lambda template: TEMPLATES[template])
    monkeypatch.setattr(spaces, '_resolutions', lambda template: ['1', '2'])
    return fetched


@pytest.mark.parametrize(
    ('name', 'frm', 'to'),
    [
        (
            'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
            'MNI152NLin2009cAsym',
            'MNI152NLin6Asym',
        ),
        (
            'tpl-MNI152NLin6Asym_from-MNIInfant+2_mode-image_xfm.h5',
            'MNIInfant+2',
            'MNI152NLin6Asym',
        ),
        (
            '/tf/tpl-MNIInfant/cohort-2/tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
            'MNI152NLin6Asym',
            'MNIInfant+2',
        ),
        ('tpl-fsLR_from-fsaverage_xfm.h5', 'fsaverage', 'fsLR'),
    ],
)
def test_parse_templateflow_xfm(name, frm, to):
    """TemplateFlow names a transform by its target template and its source."""
    xfm = transforms.parse_templateflow_xfm(name)
    assert (xfm.frm, xfm.to, xfm.source) == (frm, to, 'templateflow')
    assert xfm.path == name


@pytest.mark.parametrize(
    'name',
    [
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-points_xfm.h5',
        'tpl-MNI152NLin6Asym_res-01_T1w.nii.gz',
        'sub-01_from-ACPC_to-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.nii.gz',
    ],
)
def test_parse_templateflow_xfm_rejects(name):
    """Point-mode transforms and other files are not edges."""
    assert transforms.parse_templateflow_xfm(name) is None


def test_templateflow_edges():
    """Every image-mode transform TemplateFlow lists becomes an edge."""
    edges = {(xfm.frm, xfm.to) for xfm in transforms.templateflow_edges()}
    assert edges == {
        ('MNI152NLin6Asym', 'MNI152NLin2009cAsym'),
        ('MNI152NLin2009cAsym', 'MNI152NLin6Asym'),
        ('MNIInfant+2', 'MNI152NLin6Asym'),
        ('OASIS30ANTs', 'MNI152NLin6Asym'),
        ('MNI152NLin6Asym', 'MNIInfant+2'),
    }


def _edges(*hops):
    return [Xfm(path=f'from-{frm}_to-{to}.h5', frm=frm, to=to) for frm, to in hops]


def test_find_transform_chain():
    """The shortest chain with a transform in each direction is found."""
    edges = _edges(
        ('ACPC', 'A'),
        ('A', 'ACPC'),
        ('A', 'B'),
        ('B', 'A'),
        ('B', 'C'),
        ('C', 'B'),
        # A shortcut to C that only exists in one direction
        ('ACPC', 'C'),
        # A one-way hop to D
        ('C', 'D'),
    )
    assert transforms.find_transform_chain(edges, 'ACPC', 'A') == ['ACPC', 'A']
    assert transforms.find_transform_chain(edges, 'ACPC', 'C') == ['ACPC', 'A', 'B', 'C']

    with pytest.raises(NoTransformChainError, match='ACPC to D'):
        transforms.find_transform_chain(edges, 'ACPC', 'D')

    with pytest.raises(NoTransformChainError, match='ACPC to E'):
        transforms.find_transform_chain(edges, 'ACPC', 'E')


def test_chain_files():
    """Transforms are listed in the order ANTs takes them: the last hop first."""
    edges = _edges(('ACPC', 'A'), ('A', 'ACPC'), ('A', 'B'), ('B', 'A'))

    forward = transforms.chain_files(edges, ['ACPC', 'A', 'B'])
    assert [xfm.path for xfm in forward] == ['from-A_to-B.h5', 'from-ACPC_to-A.h5']

    inverse = transforms.chain_files(edges, ['B', 'A', 'ACPC'])
    assert [xfm.path for xfm in inverse] == ['from-A_to-ACPC.h5', 'from-B_to-A.h5']

    duplicated = [*edges, Xfm(path='run-2_from-A_to-B.h5', frm='A', to='B')]
    with pytest.raises(ValueError, match='More than one from-A_to-B transform'):
        transforms.chain_files(duplicated, ['ACPC', 'A', 'B'])


@pytest.fixture
def bids_dir(tmp_path):
    """A QSIPrep-like dataset normalized to MNI152NLin2009cAsym only."""
    skeleton = load_data('tests/skeletons/longitudinal_anat.yml')
    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
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


def _names(files):
    return [str(f).replace('\\', '/').rsplit('/', 1)[-1] for f in files]


def _forbid_templateflow(monkeypatch):
    """Fail the test if TemplateFlow's transforms are listed at all."""

    def _fail():
        raise AssertionError('TemplateFlow transforms were enumerated')

    monkeypatch.setattr(transforms, '_tf_templates', _fail)


def test_collect_direct_transforms(bids_dir, fake_templateflow, monkeypatch):
    """A space the dataset was normalized to uses its transforms, and nothing else."""
    _forbid_templateflow(monkeypatch)
    anat_data, status = _collect(bids_dir, 'MNI152NLin2009cAsym')
    assert status['template_output_space'] == 'MNI152NLin2009cAsym'
    assert status['template_transform_path'] is None
    assert isinstance(anat_data['acpc_to_template_xfm'], str)
    assert isinstance(anat_data['template_to_acpc_xfm'], str)
    assert fake_templateflow == []


def test_collect_chained_transforms(bids_dir, fake_templateflow):
    """QSIPrep with MNI152NLin2009cAsym, then QSIRecon with MNI152NLin6Asym."""
    anat_data, status = _collect(bids_dir, 'MNI152NLin6Asym:res-2mm', needs_t1w_transform=True)

    assert status['has_qsiprep_t1w_transforms'] is True
    assert status['template_output_space'] == 'MNI152NLin6Asym'
    assert status['template_transform_path'] == ['ACPC', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym']

    # ANTs applies the last transform first: ACPC -> MNI152NLin2009cAsym -> MNI152NLin6Asym
    assert _names(anat_data['acpc_to_template_xfm']) == [
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]
    # MNI152NLin6Asym -> MNI152NLin2009cAsym -> ACPC
    assert _names(anat_data['template_to_acpc_xfm']) == [
        'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5',
        'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5',
    ]
    # Only the transforms on the chain were downloaded
    assert sorted(fake_templateflow) == [
        'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]


def test_collect_chained_transforms_two_templates(bids_dir):
    """A chain may pass through more than one template, and reach a cohort."""
    anat_data, status = _collect(bids_dir, 'MNIInfant:cohort-2')

    assert status['template_output_space'] == 'MNIInfant+2'
    assert status['template_transform_path'] == [
        'ACPC',
        'MNI152NLin2009cAsym',
        'MNI152NLin6Asym',
        'MNIInfant+2',
    ]
    assert _names(anat_data['acpc_to_template_xfm']) == [
        'tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]
    assert _names(anat_data['template_to_acpc_xfm']) == [
        'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5',
        'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNIInfant+2_mode-image_xfm.h5',
    ]


def test_collect_prefers_direct_transforms(bids_dir, fake_templateflow, monkeypatch):
    """When the dataset has transforms for the space, no chain is built."""
    _forbid_templateflow(monkeypatch)
    anat_dir = bids_dir / 'sub-01' / 'anat'
    for xfm in list(anat_dir.glob('*MNI152NLin2009cAsym*_xfm.h5')):
        shutil.copy(xfm, anat_dir / xfm.name.replace('MNI152NLin2009cAsym', 'MNI152NLin6Asym'))

    anat_data, status = _collect(bids_dir, 'MNI152NLin6Asym')
    assert status['template_transform_path'] is None
    assert anat_data['acpc_to_template_xfm'].endswith(
        'sub-01_from-ACPC_to-MNI152NLin6Asym_mode-image_xfm.h5'
    )
    assert fake_templateflow == []


@pytest.mark.parametrize(
    ('token', 'reason'),
    [
        # TemplateFlow only has the transform into MNI152NLin6Asym, not out of it
        ('OASIS30ANTs', 'TemplateFlow has no transforms that link them to OASIS30ANTs'),
        # No transforms at all for this cohort
        ('MNIInfant:cohort-3', 'TemplateFlow has no transforms that link them to MNIInfant'),
        ('MNIInfant:cohort-auto', 'replace cohort-auto with a cohort label'),
    ],
)
def test_collect_unreachable_space(bids_dir, token, reason):
    """A space with no chain to it stops the build, saying what the dataset has."""
    with pytest.raises(ValueError, match='was requested') as excinfo:
        _collect(bids_dir, token)

    message = str(excinfo.value)
    assert 'It has transforms for: MNI152NLin2009cAsym.' in message
    assert reason in message


def test_collect_chain_needs_both_dataset_directions(bids_dir):
    """A chain cannot start from a dataset transform that exists in one direction only."""
    (
        bids_dir / 'sub-01' / 'anat' / 'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5'
    ).unlink()

    with pytest.raises(ValueError, match='TemplateFlow has no transforms that link them'):
        _collect(bids_dir, 'MNI152NLin6Asym')


def test_collect_no_transforms_does_not_chain(bids_dir, fake_templateflow):
    """A dataset with no template transforms has nothing to chain from."""
    for xfm in (bids_dir / 'sub-01' / 'anat').glob('*_xfm.h5'):
        xfm.unlink()

    anat_data, status = _collect(bids_dir, 'MNI152NLin6Asym')
    assert status['has_qsiprep_t1w_transforms'] is False
    assert status['template_output_space'] is None
    assert anat_data['acpc_to_template_xfm'] is None
    assert fake_templateflow == []


def test_template_mapper_takes_a_chain(tmp_path):
    """TemplateMapper accepts one transform or a chain of them."""
    from qsirecon.interfaces.scalar_mapping import TemplateMapper

    files = []
    for name in ('a.h5', 'b.h5'):
        (tmp_path / name).touch()
        files.append(str(tmp_path / name))

    assert TemplateMapper(to_template_transform=files[0]).inputs.to_template_transform == files[:1]
    assert TemplateMapper(to_template_transform=files).inputs.to_template_transform == files


def test_pyafq_rejects_a_chain():
    """pyAFQ takes one transform file, so a chained output space is an error."""
    from qsirecon.workflows.recon.pyafq import init_pyafq_wf

    inputs_dict = {
        'has_qsiprep_t1w': True,
        'template_output_space': 'MNI152NLin6Asym',
        'template_transform_path': ['ACPC', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
    }
    with pytest.raises(ValueError, match='pyAFQ takes a single template-to-anatomical'):
        init_pyafq_wf(inputs_dict=inputs_dict)
