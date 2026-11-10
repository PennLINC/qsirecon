"""Tests for choosing which copy of an atlas to use, and the transforms to warp it with."""

import json
import logging

import pytest

from qsirecon.utils import spaces, transforms
from qsirecon.utils.atlases import (
    AtlasSelectorError,
    AtlasSpec,
    collect_atlases,
    parse_atlas_token,
    parse_atlases,
)
from qsirecon.utils.transforms import ANAT_SPACE, Xfm

# The cohorts of the templates TemplateFlow knows, for checking tpl- and cohort- selectors
TF_COHORTS = {
    'MNI152NLin2009cAsym': [],
    'MNI152NLin6Asym': [],
    'MNIInfant': ['1', '2'],
    'OASIS30ANTs': [],
}

# TemplateFlow links the adult templates to each other, and each infant cohort to
# MNI152NLin6Asym only.
TF_XFMS = {
    'MNI152NLin2009cAsym': ['tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5'],
    'MNI152NLin6Asym': [
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNIInfant+1_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNIInfant+2_mode-image_xfm.h5',
    ],
    'MNIInfant': [
        'tpl-MNIInfant_cohort-1_from-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
    ],
}


@pytest.fixture(autouse=True)
def fake_templateflow(monkeypatch, tmp_path):
    """Answer TemplateFlow queries from TF_XFMS, and record what is listed and downloaded."""
    tf_home = tmp_path / 'templateflow'
    calls = {'listed': 0, 'fetched': []}

    def _templates():
        calls['listed'] += 1
        return list(TF_XFMS)

    def _ls(template):
        return [str(tf_home / f'tpl-{template}' / name) for name in TF_XFMS[template]]

    def _get(template, **query):
        cohort = f'_cohort-{query["cohort"]}' if 'cohort' in query else ''
        mode = f'_mode-{query["mode"]}' if 'mode' in query else ''
        name = f'tpl-{template}{cohort}_from-{query["from"]}{mode}_xfm.h5'
        if name not in TF_XFMS[template]:
            return []

        calls['fetched'].append(name)
        return tf_home / f'tpl-{template}' / name

    monkeypatch.setattr(transforms, '_tf_templates', _templates)
    monkeypatch.setattr(transforms, '_tf_ls', _ls)
    monkeypatch.setattr(transforms, '_tf_get', _get)
    monkeypatch.setattr(spaces, '_templates', lambda: list(TF_COHORTS))
    monkeypatch.setattr(spaces, '_cohorts', lambda template: TF_COHORTS[template])
    return calls


def _atlas_dataset(root, copies, cohort_in_filename=True, voxel_sizes=None):
    """Write a BIDS-Atlas dataset with empty images.

    ``copies`` maps an atlas name to the spaces it is available in.
    A space may be followed by the other entities of the file, written as selectors
    (``MNI152NLin2009cAsym:res-01:desc-LR``).
    BIDS-Atlas allows a cohort to be given by its folder alone, which
    ``cohort_in_filename=False`` writes.
    ``voxel_sizes`` maps a ``res`` label to a voxel size in millimeters; images with
    such a label are written as real NIfTI files rather than empty ones.
    """
    import nibabel as nb
    import numpy as np

    root.mkdir(parents=True)
    description = {'Name': 'Test atlases', 'BIDSVersion': '1.9.0', 'DatasetType': 'derivative'}
    (root / 'dataset_description.json').write_text(json.dumps(description))
    for atlas, copies_of_atlas in copies.items():
        for copy in copies_of_atlas:
            space, *extra = copy.split(':')
            template, _, cohort = space.partition('+')
            folder = root / f'tpl-{template}'
            prefix = f'tpl-{template}'
            if cohort:
                folder = folder / f'cohort-{cohort}'
                if cohort_in_filename:
                    prefix += f'_cohort-{cohort}'

            folder.mkdir(parents=True, exist_ok=True)
            stem = '_'.join([prefix, f'atlas-{atlas}', *extra, 'dseg'])
            res = dict(item.split('-') for item in extra).get('res')
            if voxel_sizes and res in voxel_sizes:
                affine = np.diag([voxel_sizes[res]] * 3 + [1])
                img = nb.Nifti1Image(np.zeros((2, 2, 2), dtype='uint8'), affine)
                img.to_filename(folder / f'{stem}.nii.gz')
            else:
                (folder / f'{stem}.nii.gz').touch()

            (folder / f'{stem}.tsv').write_text('index\tname\n1\tone\n2\ttwo\n')

    return {'testatlases': str(root)}


def _normalized_to(space):
    """The template transforms of a dataset normalized to ``space``, in both directions."""
    return [
        Xfm(path=f'/data/sub-01_from-ACPC_to-{space}_mode-image_xfm.h5', frm=ANAT_SPACE, to=space),
        Xfm(path=f'/data/sub-01_from-{space}_to-ACPC_mode-image_xfm.h5', frm=space, to=ANAT_SPACE),
    ]


def _names(files):
    return [str(f).replace('\\', '/').rsplit('/', 1)[-1] for f in files]


def test_collect_atlases_without_transforms(tmp_path, fake_templateflow):
    """Without transforms to choose by, atlases come from MNI152NLin2009cAsym."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Both': ['MNI152NLin2009cAsym', 'MNIInfant+2']},
    )

    configs = collect_atlases(datasets=datasets, atlases=['Both'])
    assert configs['Both']['space'] == 'MNI152NLin2009cAsym'
    assert 'xfm_to_anat' not in configs['Both']
    assert fake_templateflow == {'listed': 0, 'fetched': []}


@pytest.mark.parametrize(
    ('normalized_to', 'expected'),
    [
        ('MNI152NLin2009cAsym', 'MNI152NLin2009cAsym'),
        # Infant data use the infant copy of an atlas that has one
        ('MNIInfant+2', 'MNIInfant+2'),
    ],
)
def test_collect_atlases_prefers_direct_copy(tmp_path, fake_templateflow, normalized_to, expected):
    """The copy in the template the data were normalized to is used, with one transform."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Both': ['MNI152NLin2009cAsym', 'MNIInfant+2']},
    )

    configs = collect_atlases(
        datasets=datasets,
        atlases=['Both'],
        xfm_edges=_normalized_to(normalized_to),
    )
    config = configs['Both']
    assert config['space'] == expected
    assert config['xfm_path'] == [expected, 'ACPC']
    assert _names(config['xfm_to_anat']) == [f'sub-01_from-{expected}_to-ACPC_mode-image_xfm.h5']
    assert f'tpl-{expected.partition("+")[0]}' in config['image']
    assert config['labels'].endswith('_dseg.tsv')
    # TemplateFlow is not consulted when a direct copy exists
    assert fake_templateflow == {'listed': 0, 'fetched': []}


def test_collect_atlases_cohort_from_folder(tmp_path, fake_templateflow):
    """A cohort named only by the atlas's folder is still recognized."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Folder': ['MNI152NLin2009cAsym', 'MNIInfant+1', 'MNIInfant+2']},
        cohort_in_filename=False,
    )

    configs = collect_atlases(
        datasets=datasets,
        atlases=['Folder'],
        xfm_edges=_normalized_to('MNIInfant+2'),
    )
    config = configs['Folder']
    assert config['space'] == 'MNIInfant+2'
    assert config['xfm_path'] == ['MNIInfant+2', 'ACPC']
    assert 'cohort-2' in config['image'].replace('\\', '/')
    assert fake_templateflow == {'listed': 0, 'fetched': []}


def test_collect_atlases_chains_adult_atlas_into_infant_data(tmp_path, fake_templateflow, caplog):
    """An atlas with no infant copy is chained through TemplateFlow, with a warning."""
    datasets = _atlas_dataset(tmp_path / 'atlases', {'AdultOnly': ['MNI152NLin2009cAsym']})

    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        configs = collect_atlases(
            datasets=datasets,
            atlases=['AdultOnly'],
            xfm_edges=_normalized_to('MNIInfant+2'),
        )

    config = configs['AdultOnly']
    assert config['space'] == 'MNI152NLin2009cAsym'
    assert config['xfm_path'] == [
        'MNI152NLin2009cAsym',
        'MNI152NLin6Asym',
        'MNIInfant+2',
        'ACPC',
    ]
    # ANTs applies the last transform first
    assert _names(config['xfm_to_anat']) == [
        'sub-01_from-MNIInfant+2_to-ACPC_mode-image_xfm.h5',
        'tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ]
    assert sorted(fake_templateflow['fetched']) == [
        'tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        'tpl-MNIInfant_cohort-2_from-MNI152NLin6Asym_mode-image_xfm.h5',
    ]
    warning = caplog.text
    assert 'The atlas AdultOnly is in MNI152NLin2009cAsym' in warning
    assert 'MNI152NLin2009cAsym -> MNI152NLin6Asym -> MNIInfant+2 -> ACPC' in warning


def test_collect_atlases_tie_prefers_same_template(tmp_path):
    """Between equally long chains, another cohort of the dataset's template wins."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'OtherCohort': ['MNI152NLin2009cAsym', 'MNIInfant+1']},
    )

    configs = collect_atlases(
        datasets=datasets,
        atlases=['OtherCohort'],
        xfm_edges=_normalized_to('MNIInfant+2'),
    )
    config = configs['OtherCohort']
    assert config['space'] == 'MNIInfant+1'
    assert config['xfm_path'] == ['MNIInfant+1', 'MNI152NLin6Asym', 'MNIInfant+2', 'ACPC']


def test_collect_atlases_per_atlas_templates(tmp_path):
    """Atlases requested together may come from different templates."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Adult': ['MNI152NLin2009cAsym'], 'Six': ['MNI152NLin6Asym']},
    )

    configs = collect_atlases(
        datasets=datasets,
        atlases=['Adult', 'Six'],
        xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
    )
    assert configs['Adult']['xfm_path'] == ['MNI152NLin2009cAsym', 'ACPC']
    assert configs['Six']['xfm_path'] == ['MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'ACPC']
    assert _names(configs['Six']['xfm_to_anat']) == [
        'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5',
        'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5',
    ]


def test_collect_atlases_template_filter(tmp_path):
    """A template in the BIDS filters restricts where an atlas may be taken from."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Both': ['MNI152NLin2009cAsym', 'MNI152NLin6Asym']},
    )

    configs = collect_atlases(
        datasets=datasets,
        atlases=['Both'],
        bids_filters={'atlas': {'template': 'MNI152NLin6Asym'}},
        xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
    )
    assert configs['Both']['space'] == 'MNI152NLin6Asym'
    assert len(configs['Both']['xfm_to_anat']) == 2


def test_collect_atlases_unreachable(tmp_path):
    """An atlas that no chain of transforms reaches stops the build."""
    datasets = _atlas_dataset(tmp_path / 'atlases', {'Odd': ['OASIS30ANTs']})

    with pytest.raises(ValueError, match='No chain of transforms brings the atlas Odd') as excinfo:
        collect_atlases(
            datasets=datasets,
            atlases=['Odd'],
            xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
        )

    assert 'available in OASIS30ANTs' in str(excinfo.value)
    assert 'transforms from: MNI152NLin2009cAsym' in str(excinfo.value)


def test_collect_atlases_missing(tmp_path):
    """An atlas that is in no dataset is reported."""
    datasets = _atlas_dataset(tmp_path / 'atlases', {'Present': ['MNI152NLin2009cAsym']})

    with pytest.raises(ValueError, match='No atlas images found for Absent'):
        collect_atlases(
            datasets=datasets,
            atlases=['Present', 'Absent'],
            xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
        )


@pytest.mark.parametrize(
    ('token', 'expected', 'canonical'),
    [
        ('4S156Parcels', AtlasSpec(name='4S156Parcels'), '4S156Parcels'),
        (
            '4S156Parcels:tpl-MNI152NLin2009cAsym:res-01',
            AtlasSpec(name='4S156Parcels', tpl='MNI152NLin2009cAsym', res='01'),
            '4S156Parcels:tpl-MNI152NLin2009cAsym:res-01',
        ),
        # Selectors are written out in a fixed order
        (
            'Mine:seg-x:desc-LR:res-1:cohort-2:tpl-MNIInfant',
            AtlasSpec(name='Mine', tpl='MNIInfant', cohort='2', res='1', desc='LR', seg='x'),
            'Mine:tpl-MNIInfant:cohort-2:res-1:desc-LR:seg-x',
        ),
        # A template TemplateFlow does not know may be one the dataset has transforms for
        ('Mine:tpl-Custom:cohort-A', AtlasSpec(name='Mine', tpl='Custom', cohort='A'), None),
        # A resolution needs no template
        ('Mine:res-02', AtlasSpec(name='Mine', res='02'), None),
    ],
)
def test_parse_atlas_token(token, expected, canonical):
    """A token is an atlas name followed by selectors, and round-trips through str."""
    spec = parse_atlas_token(token)
    assert spec == expected
    assert str(spec) == (canonical or token)
    assert parse_atlas_token(str(spec)) == spec


def test_atlas_spec_selectors():
    """Selectors are matched literally, and only the ones that were given."""
    spec = parse_atlas_token('Mine:tpl-MNI152NLin2009cAsym:res-01')
    assert spec.selectors == {'tpl': 'MNI152NLin2009cAsym', 'res': '01'}
    assert spec.selector_str == 'tpl-MNI152NLin2009cAsym:res-01'
    assert spec.matches({'tpl': 'MNI152NLin2009cAsym', 'res': '01', 'desc': 'LR'})
    assert not spec.matches({'tpl': 'MNI152NLin2009cAsym', 'res': '1'})
    assert not spec.matches({'tpl': 'MNI152NLin2009cAsym'})
    assert parse_atlas_token('Mine').selector_str == ''
    assert parse_atlas_token('Mine').matches({'tpl': 'MNIInfant', 'cohort': '2'})


@pytest.mark.parametrize(
    ('token', 'message'),
    [
        (':tpl-MNI152NLin2009cAsym', 'does not start with the name of an atlas'),
        ('Mine:MNI152NLin2009cAsym', 'is not a key-value specification'),
        ('Mine:res-', 'is not a key-value specification'),
        ('Mine:den-91k', '"den" in "Mine:den-91k" is not a valid specifier'),
        ('Mine:space-MNI152NLin2009cAsym', 'QSIRecon accepts: tpl, cohort, res, desc, seg'),
        ('Mine:res-01:res-02', 'specifies more than one res-'),
        ('Mine:desc-a_b', 'a label may only have letters and digits'),
        ('Mine:res-2mm', 'res- only selects the atlas file'),
        ('Mine:res-1p5x1p5x3mm', 'res- only selects the atlas file'),
        ('Mine:cohort-2', 'names a cohort without a template'),
        ('Mine:tpl-MNIInfant', 'it needs a cohort. Use one of: cohort-1, cohort-2'),
        ('Mine:tpl-MNIInfant:cohort-auto', 'cohort-auto in .* is not valid in --atlases'),
        ('Mine:tpl-MNIInfant:cohort-9', 'has no cohort-9'),
        ('Mine:tpl-MNI152NLin2009cAsym:cohort-1', 'does not accept a cohort'),
    ],
)
def test_parse_atlas_token_errors(token, message):
    """Malformed selectors are rejected with a message that says what is wrong."""
    with pytest.raises(AtlasSelectorError, match=message):
        parse_atlas_token(token)


def test_parse_atlases_duplicates():
    """An atlas may be requested once; an exact repeat is the same request."""
    specs = parse_atlases(['A', 'B:res-01', 'A', 'B:res-01'])
    assert [str(spec) for spec in specs] == ['A', 'B:res-01']

    with pytest.raises(AtlasSelectorError, match='The atlas A was requested more than once'):
        parse_atlases(['A:tpl-MNI152NLin2009cAsym', 'A:tpl-MNI152NLin6Asym'])

    with pytest.raises(AtlasSelectorError, match='seg-A'):
        parse_atlases(['A', 'A:res-01'])


def test_collect_atlases_selector_pins_template(tmp_path, fake_templateflow, caplog):
    """A tpl- selector uses that copy, even when another has a shorter chain."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {
            'Pinned': ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
            'Free': ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
        },
    )

    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        configs = collect_atlases(
            datasets=datasets,
            atlases=['Pinned:tpl-MNI152NLin6Asym', 'Free'],
            xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
        )

    # The configs are keyed by the bare atlas names
    assert sorted(configs) == ['Free', 'Pinned']
    pinned = configs['Pinned']
    assert pinned['space'] == 'MNI152NLin6Asym'
    assert 'tpl-MNI152NLin6Asym_atlas-Pinned_dseg.nii.gz' in pinned['image']
    assert 'tpl-MNI152NLin6Asym_atlas-Pinned_dseg.tsv' in pinned['labels']
    assert pinned['xfm_path'] == ['MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'ACPC']
    assert _names(pinned['xfm_to_anat']) == [
        'sub-01_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5',
        'tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5',
    ]
    # A chained atlas is still warned about when it was asked for
    assert 'The atlas Pinned is in MNI152NLin6Asym' in caplog.text
    # The atlas without selectors keeps the automatic choice
    assert configs['Free']['xfm_path'] == ['MNI152NLin2009cAsym', 'ACPC']
    assert 'The atlas Free' not in caplog.text


def test_collect_atlases_selector_pins_cohort(tmp_path, fake_templateflow):
    """A cohort selector matches a cohort that only the atlas's folder names."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Folder': ['MNI152NLin2009cAsym', 'MNIInfant+1', 'MNIInfant+2']},
        cohort_in_filename=False,
    )

    configs = collect_atlases(
        datasets=datasets,
        atlases=['Folder:tpl-MNIInfant:cohort-1'],
        xfm_edges=_normalized_to('MNIInfant+2'),
    )
    config = configs['Folder']
    assert config['space'] == 'MNIInfant+1'
    assert config['xfm_path'] == ['MNIInfant+1', 'MNI152NLin6Asym', 'MNIInfant+2', 'ACPC']
    assert 'cohort-1' in config['image'].replace('\\', '/')


def test_collect_atlases_selector_pins_resolution(tmp_path, caplog):
    """A res- selector chooses between the resolutions of one template, without a warning."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {
            'Multi': [
                'MNI152NLin2009cAsym:res-01',
                'MNI152NLin2009cAsym:res-02',
                'MNI152NLin2009cAsym:res-02:desc-LR',
                'MNI152NLin6Asym:res-02',
            ],
        },
    )

    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        configs = collect_atlases(
            datasets=datasets,
            atlases=['Multi:tpl-MNI152NLin2009cAsym:res-02:desc-LR'],
            xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
        )

    image = configs['Multi']['image']
    assert image.endswith('tpl-MNI152NLin2009cAsym_atlas-Multi_res-02_desc-LR_dseg.nii.gz')
    assert configs['Multi']['labels'].endswith('_res-02_desc-LR_dseg.tsv')
    assert caplog.text == ''

    # A resolution alone leaves the template to the transform chains
    configs = collect_atlases(
        datasets=datasets,
        atlases=['Multi:res-01'],
        xfm_edges=_normalized_to('MNI152NLin6Asym'),
    )
    image = configs['Multi']['image']
    assert image.endswith('tpl-MNI152NLin2009cAsym_atlas-Multi_res-01_dseg.nii.gz')
    assert configs['Multi']['xfm_path'] == ['MNI152NLin2009cAsym', 'MNI152NLin6Asym', 'ACPC']


@pytest.mark.parametrize('atlases', [['Multi'], ['Multi:tpl-MNI152NLin2009cAsym']])
def test_collect_atlases_highest_resolution(tmp_path, caplog, atlases):
    """With several resolutions and no res- selector, the smallest voxels win, with a warning."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Multi': ['MNI152NLin2009cAsym:res-01', 'MNI152NLin2009cAsym:res-02']},
        # The labels say nothing about the voxel sizes: the images are read
        voxel_sizes={'01': 2, '02': 1},
    )

    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        configs = collect_atlases(
            datasets=datasets,
            atlases=atlases,
            xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
        )

    assert configs['Multi']['image'].endswith('_atlas-Multi_res-02_dseg.nii.gz')
    assert 'The atlas Multi is available at more than one resolution' in caplog.text
    assert 'add a res- selector' in caplog.text


def test_collect_atlases_selector_without_match(tmp_path):
    """Selectors that no copy of the atlas has stop the build, listing the copies there are."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Multi': ['MNI152NLin2009cAsym:res-01', 'MNIInfant+2:res-01'], 'Other': ['MNIInfant+1']},
    )

    with pytest.raises(ValueError, match='Errors found in atlas collection') as excinfo:
        collect_atlases(
            datasets=datasets,
            atlases=['Multi:tpl-MNI152NLin2009cAsym:res-1', 'Other:tpl-MNIInfant:cohort-2'],
            xfm_edges=_normalized_to('MNI152NLin2009cAsym'),
        )

    message = str(excinfo.value)
    assert (
        'No copy of the atlas Multi matches the selectors "tpl-MNI152NLin2009cAsym:res-1"'
    ) in message
    assert 'are: tpl-MNI152NLin2009cAsym:res-01, tpl-MNIInfant:cohort-2:res-01.' in message
    assert 'No copy of the atlas Other matches the selectors "tpl-MNIInfant:cohort-2"' in message
    assert 'are: tpl-MNIInfant:cohort-1.' in message


def test_collect_atlases_selector_unreachable(tmp_path):
    """A pinned copy that no chain reaches is an error, even if another copy is reachable."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Both': ['MNI152NLin2009cAsym', 'OASIS30ANTs']},
    )
    xfm_edges = _normalized_to('MNI152NLin2009cAsym')

    with pytest.raises(
        ValueError, match='No chain of transforms brings the atlas Both'
    ) as excinfo:
        collect_atlases(
            datasets=datasets,
            atlases=['Both:tpl-OASIS30ANTs'],
            xfm_edges=xfm_edges,
        )

    message = str(excinfo.value)
    assert 'requested with the selectors "tpl-OASIS30ANTs"' in message
    assert 'the copies that match are in OASIS30ANTs' in message
    assert 'transforms from: MNI152NLin2009cAsym' in message

    configs = collect_atlases(datasets=datasets, atlases=['Both'], xfm_edges=xfm_edges)
    assert configs['Both']['space'] == 'MNI152NLin2009cAsym'


def test_collect_atlases_selector_and_bids_filter(tmp_path):
    """The selectors of an atlas and the atlas BIDS filter must agree."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Both': ['MNI152NLin2009cAsym', 'MNI152NLin6Asym']},
    )
    xfm_edges = _normalized_to('MNI152NLin2009cAsym')

    with pytest.raises(ValueError, match='The two must agree') as excinfo:
        collect_atlases(
            datasets=datasets,
            atlases=['Both:tpl-MNI152NLin6Asym'],
            bids_filters={'atlas': {'template': 'MNI152NLin2009cAsym'}},
            xfm_edges=xfm_edges,
        )

    message = str(excinfo.value)
    assert 'The atlas Both was requested with tpl-MNI152NLin6Asym' in message
    assert 'sets "template" to MNI152NLin2009cAsym' in message

    configs = collect_atlases(
        datasets=datasets,
        atlases=['Both:tpl-MNI152NLin6Asym'],
        bids_filters={'atlas': {'template': ['MNI152NLin2009cAsym', 'MNI152NLin6Asym']}},
        xfm_edges=xfm_edges,
    )
    assert configs['Both']['space'] == 'MNI152NLin6Asym'


def test_collect_atlases_selector_without_transforms(tmp_path):
    """Without transforms to choose by, a tpl- selector replaces the default template."""
    datasets = _atlas_dataset(
        tmp_path / 'atlases',
        {'Both': ['MNI152NLin2009cAsym', 'MNIInfant+2']},
    )

    configs = collect_atlases(datasets=datasets, atlases=['Both:tpl-MNIInfant:cohort-2'])
    assert configs['Both']['space'] == 'MNIInfant+2'
    assert 'xfm_to_anat' not in configs['Both']


def test_collect_atlases_rejects_bad_tokens(tmp_path):
    """Tokens that did not come through the command line are still validated."""
    datasets = _atlas_dataset(tmp_path / 'atlases', {'Both': ['MNI152NLin2009cAsym']})

    with pytest.raises(AtlasSelectorError, match='requested more than once'):
        collect_atlases(datasets=datasets, atlases=['Both', 'Both:res-01'])


def test_warp_atlases_wf_uses_each_atlas_transforms(tmp_path):
    """Each atlas is warped with its own transforms."""
    from qsirecon.workflows.recon.anatomical import init_warp_atlases_wf

    files = {}
    for name in ('a_dseg.nii.gz', 'a_dseg.tsv', 'b_dseg.nii.gz', 'b_dseg.tsv', 'x.h5', 'y.h5'):
        files[name] = str(tmp_path / name)
        open(files[name], 'w').close()

    atlas_configs = {
        'B': {
            'image': files['b_dseg.nii.gz'],
            'labels': files['b_dseg.tsv'],
            'xfm_to_anat': [files['x.h5'], files['y.h5']],
            'xfm_path': ['MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'ACPC'],
        },
        'A': {
            'image': files['a_dseg.nii.gz'],
            'labels': files['a_dseg.tsv'],
            'xfm_to_anat': [files['x.h5']],
            'xfm_path': ['MNI152NLin2009cAsym', 'ACPC'],
        },
    }
    wf = init_warp_atlases_wf(atlas_configs=atlas_configs)

    warp_atlases = wf.get_node('warp_atlases')
    assert 'transforms' in warp_atlases.iterfield
    # In the order of the sorted atlas names
    assert warp_atlases.inputs.transforms == [[files['x.h5']], [files['x.h5'], files['y.h5']]]
    assert wf.get_node('conform_atlases').inputs.in_file == [
        files['a_dseg.nii.gz'],
        files['b_dseg.nii.gz'],
    ]
    assert 'The B atlas is defined in MNI152NLin6Asym' in wf.__desc__
    assert 'The A atlas is defined' not in wf.__desc__
