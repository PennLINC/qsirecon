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
        # Into MNI152NLin6Asym only, so OASIS30ANTs cannot be an output space
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


def _forbid_templateflow(monkeypatch):
    """Fail the test if TemplateFlow's transforms are listed at all."""

    def _fail():
        raise AssertionError('TemplateFlow transforms were enumerated')

    monkeypatch.setattr(transforms, '_tf_templates', _fail)


def test_collect_direct_transforms(bids_dir, fake_templateflow, monkeypatch):
    """A space the dataset was normalized to uses its transforms, and nothing else."""
    _forbid_templateflow(monkeypatch)
    _, status = _collect(bids_dir, 'MNI152NLin2009cAsym')
    (space_xfms,) = status['output_space_xfms']
    assert space_xfms['space'] == 'MNI152NLin2009cAsym'
    assert space_xfms['transform_path'] is None
    assert _names(space_xfms['acpc_to_template_xfm']) == [
        'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]
    assert fake_templateflow == []


def test_collect_chained_transforms(bids_dir, fake_templateflow):
    """QSIPrep with MNI152NLin2009cAsym, then QSIRecon with MNI152NLin6Asym."""
    _, status = _collect(bids_dir, 'MNI152NLin6Asym:res-2mm', needs_anat_transform=True)

    assert status['has_qsiprep_anat_transforms'] is True
    (space_xfms,) = status['output_space_xfms']
    assert space_xfms['token'] == 'MNI152NLin6Asym:res-2mm'
    assert space_xfms['space'] == 'MNI152NLin6Asym'
    assert space_xfms['transform_path'] == ['ACPC', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym']

    # ANTs applies the last transform first: ACPC -> MNI152NLin2009cAsym -> MNI152NLin6Asym
    assert _names(space_xfms['acpc_to_template_xfm']) == [
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]
    # Only the transform on the chain was downloaded, and only in the direction used
    assert fake_templateflow == ['tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5']


def test_collect_direct_and_chained_spaces_together(bids_dir, fake_templateflow):
    """Each requested space is resolved on its own: one directly, one through a chain."""
    _, status = _collect(
        bids_dir,
        'MNI152NLin2009cAsym:res-2mm',
        'MNI152NLin6Asym:res-1:res-2mm',
    )

    direct, chained_1, chained_2 = status['output_space_xfms']
    assert direct['token'] == 'MNI152NLin2009cAsym:res-2mm'
    assert direct['transform_path'] is None
    assert _names(direct['acpc_to_template_xfm']) == [
        'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]

    assert [chained_1['token'], chained_2['token']] == [
        'MNI152NLin6Asym:res-1',
        'MNI152NLin6Asym:res-2mm',
    ]
    for chained in (chained_1, chained_2):
        assert chained['transform_path'] == ['ACPC', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym']
        assert _names(chained['acpc_to_template_xfm']) == [
            'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
            'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
        ]

    # The two resolutions share one chain, so its transform was fetched once
    assert fake_templateflow == ['tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5']


def test_collect_chained_transforms_two_templates(bids_dir):
    """A chain may pass through more than one template, and reach a cohort."""
    _, status = _collect(bids_dir, 'MNIInfant:cohort-2')

    (space_xfms,) = status['output_space_xfms']
    assert space_xfms['space'] == 'MNIInfant+2'
    assert space_xfms['transform_path'] == [
        'ACPC',
        'MNI152NLin2009cAsym',
        'MNI152NLin6Asym',
        'MNIInfant+2',
    ]
    assert _names(space_xfms['acpc_to_template_xfm']) == [
        'tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]


def test_collect_prefers_direct_transforms(bids_dir, fake_templateflow, monkeypatch):
    """When the dataset has transforms for the space, no chain is built."""
    _forbid_templateflow(monkeypatch)
    anat_dir = bids_dir / 'sub-01' / 'anat'
    for xfm in list(anat_dir.glob('*MNI152NLin2009cAsym*_xfm.h5')):
        shutil.copy(xfm, anat_dir / xfm.name.replace('MNI152NLin2009cAsym', 'MNI152NLin6Asym'))

    _, status = _collect(bids_dir, 'MNI152NLin6Asym')
    (space_xfms,) = status['output_space_xfms']
    assert space_xfms['transform_path'] is None
    assert _names(space_xfms['acpc_to_template_xfm']) == [
        'sub-01_from-ACPC_to-MNI152NLin6Asym_mode-image_xfm.h5',
    ]
    assert fake_templateflow == []


@pytest.mark.parametrize(
    ('token', 'reason'),
    [
        # TemplateFlow only has the transform out of OASIS30ANTs, not into it
        ('OASIS30ANTs', 'TemplateFlow has no transforms that link them to OASIS30ANTs'),
        # No transforms at all for this cohort
        ('MNIInfant:cohort-3', 'TemplateFlow has no transforms that link them to MNIInfant'),
        # cohort-auto only looks at the dataset's own transforms
        ('MNIInfant:cohort-auto', 'replace cohort-auto with a cohort label'),
    ],
)
def test_collect_unreachable_space(bids_dir, token, reason):
    """A space with no chain to it stops the build, saying what the dataset has."""
    with pytest.raises(ValueError, match='was requested') as excinfo:
        _collect(bids_dir, token)

    message = str(excinfo.value)
    assert 'It has transforms to: MNI152NLin2009cAsym.' in message
    assert reason in message


def test_collect_chain_is_one_way(bids_dir, fake_templateflow):
    """A chain to an output space only needs the transforms that go toward the template."""
    anat_dir = bids_dir / 'sub-01' / 'anat'
    (anat_dir / 'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5').unlink()

    _, status = _collect(bids_dir, 'MNI152NLin6Asym')
    (space_xfms,) = status['output_space_xfms']
    assert space_xfms['transform_path'] == ['ACPC', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym']
    assert fake_templateflow == ['tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5']
    # The ODF report ROIs need the direction that is missing, so they are left out
    assert status['odf_rois_to_acpc_xfm'] is None


def test_collect_chain_can_require_both_directions(bids_dir, monkeypatch):
    """The one-way search is a setting: the chains can be made to need both directions."""
    monkeypatch.setattr(xbids, 'OUTPUT_SPACE_CHAINS_BIDIRECTIONAL', True)
    anat_dir = bids_dir / 'sub-01' / 'anat'
    (anat_dir / 'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5').unlink()

    with pytest.raises(ValueError, match='TemplateFlow has no transforms that link them'):
        _collect(bids_dir, 'MNI152NLin6Asym')


def test_collect_only_inverse_transforms(bids_dir, fake_templateflow):
    """Template-to-anatomical transforms alone cannot write anything in a template."""
    anat_dir = bids_dir / 'sub-01' / 'anat'
    (anat_dir / 'sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5').unlink()

    _, status = _collect(bids_dir, 'MNI152NLin2009cAsym')
    assert status['has_qsiprep_anat_transforms'] is False
    assert status['output_space_xfms'] == []
    # The edges are still there for atlases to be chosen with
    assert [(xfm.frm, xfm.to) for xfm in status['template_xfm_edges']] == [
        ('MNI152NLin2009cAsym', 'ACPC'),
    ]
    assert fake_templateflow == []


def test_collect_no_transforms_does_not_chain(bids_dir, fake_templateflow):
    """A dataset with no template transforms has nothing to chain from."""
    for xfm in (bids_dir / 'sub-01' / 'anat').glob('*_xfm.h5'):
        xfm.unlink()

    _, status = _collect(bids_dir, 'MNI152NLin6Asym')
    assert status['has_qsiprep_anat_transforms'] is False
    assert status['output_space_xfms'] == []
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


def test_find_transform_chain_one_way():
    """A one-way search only needs the files for the direction asked for."""
    edges = _edges(('A', 'B'), ('B', 'ACPC'), ('ACPC', 'B'))

    with pytest.raises(NoTransformChainError):
        transforms.find_transform_chain(edges, 'A', 'ACPC')

    assert transforms.find_transform_chain(edges, 'A', 'ACPC', bidirectional=False) == [
        'A',
        'B',
        'ACPC',
    ]
    with pytest.raises(NoTransformChainError):
        transforms.find_transform_chain(edges, 'ACPC', 'A', bidirectional=False)


def _infant_only(bids_dir):
    """Rename the dataset's template transforms so it is normalized to MNIInfant+2 only."""
    anat_dir = bids_dir / 'sub-01' / 'anat'
    for xfm in list(anat_dir.glob('*MNI152NLin2009cAsym*_xfm.h5')):
        xfm.rename(anat_dir / xfm.name.replace('MNI152NLin2009cAsym', 'MNIInfant+2'))


@pytest.mark.parametrize('token', ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'])
def test_odf_rois_use_dataset_transform(bids_dir, fake_templateflow, token):
    """The ODF ROIs come from MNI152NLin2009cAsym, whatever the output space is."""
    _, status = _collect(bids_dir, token)
    assert _names(status['odf_rois_to_acpc_xfm']) == [
        'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5',
    ]
    assert not any('MNIInfant' in name for name in fake_templateflow)


def test_odf_rois_chained_for_infant_dataset(bids_dir, fake_templateflow):
    """A dataset normalized to an infant template reaches the ROIs through TemplateFlow."""
    _infant_only(bids_dir)

    _, status = _collect(bids_dir, 'MNIInfant:cohort-auto')
    (space_xfms,) = status['output_space_xfms']
    assert space_xfms['space'] == 'MNIInfant+2'
    # The output space itself is direct
    assert space_xfms['transform_path'] is None
    # MNI152NLin2009cAsym -> MNI152NLin6Asym -> MNIInfant+2 -> ACPC, last hop first
    assert _names(status['odf_rois_to_acpc_xfm']) == [
        'sub-01_from-MNIInfant+2_to-ACPC_mode-image_xfm.h5',
        'tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]
    assert sorted(fake_templateflow) == [
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
    ]


def test_odf_rois_dropped_when_download_fails(bids_dir, monkeypatch):
    """A failed download does not stop the build; the ODF figures are just not made."""
    _infant_only(bids_dir)

    def _offline(template, **query):
        raise OSError('no network')

    monkeypatch.setattr(transforms, '_tf_get', _offline)

    _, status = _collect(bids_dir, 'MNIInfant:cohort-2')
    assert status['has_qsiprep_anat_transforms'] is True
    assert len(status['output_space_xfms']) == 1
    assert status['odf_rois_to_acpc_xfm'] is None


def test_odf_rois_skipped_without_reports(bids_dir, monkeypatch):
    """With --skip-odf-reports, TemplateFlow is not consulted for the ROIs."""
    from qsirecon import config

    _infant_only(bids_dir)
    _forbid_templateflow(monkeypatch)
    monkeypatch.setattr(config.execution, 'skip_odf_reports', True)

    _, status = _collect(bids_dir, 'MNIInfant:cohort-2')
    assert status['odf_rois_to_acpc_xfm'] is None


@pytest.mark.parametrize('chained', [False, True])
def test_dwi_anatomical_wf_odf_rois(tmp_path, monkeypatch, chained):
    """The ODF ROI node takes its own transforms when they were found."""
    from qsirecon import config
    from qsirecon.workflows.recon.anatomical import init_dwi_recon_anatomical_workflow

    monkeypatch.setattr(config.execution, 'output_dir', tmp_path)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.workflow, 'output_spaces', ['MNI152NLin6Asym'])

    odf_xfm = None
    if chained:
        odf_xfm = [
            str(tmp_path / 'from-A_to-ACPC_xfm.h5'),
            str(tmp_path / 'tpl-A_from-MNI152NLin2009cAsym_xfm.h5'),
        ]
        for xfm in odf_xfm:
            open(xfm, 'w').close()

    wf, status = init_dwi_recon_anatomical_workflow(
        atlas_configs={},
        has_qsiprep_5tt_hsvs=False,
        needs_anat_transform=False,
        has_freesurfer_5tt_hsvs=False,
        has_qsiprep_anat=True,
        has_qsiprep_anat_transforms=True,
        has_freesurfer=False,
        extras_to_make=[],
        name='dwi_specific_anat_wf',
        odf_rois_to_acpc_xfm=odf_xfm,
    )

    assert status['has_odf_rois'] is chained
    odf_rois = wf.get_node('odf_rois')
    if chained:
        assert odf_rois.inputs.transforms == odf_xfm
        assert odf_rois.inputs.input_image.endswith(
            'tpl-MNI152NLin2009cAsym_atlas-crossingfibers_res-01_dseg.nii.gz'
        )
    else:
        # Without transforms for the ROIs they are not warped, and no ODF figure is made
        assert odf_rois is None


def test_odf_roi_file():
    """The ROI image and its labels table agree, and sit on the template's 1 mm grid."""
    import nibabel as nb
    import numpy as np
    import pandas as pd

    stem = 'tpl-MNI152NLin2009cAsym_atlas-crossingfibers_res-01_dseg'
    img = nb.load(load_data(f'{stem}.nii.gz'))
    labels = pd.read_table(load_data(f'{stem}.tsv'))

    assert img.shape == (193, 229, 193)
    assert img.header.get_zooms() == (1.0, 1.0, 1.0)
    found = sorted(int(value) for value in np.unique(np.asanyarray(img.dataobj)) if value)
    assert found == labels['index'].tolist() == [1, 2, 3]
    assert labels['name'].notna().all()
