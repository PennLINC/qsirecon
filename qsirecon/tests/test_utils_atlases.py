"""Tests for choosing which copy of an atlas to use, and the transforms to warp it with."""

import json
import logging

import pytest

from qsirecon.utils import transforms
from qsirecon.utils.atlases import collect_atlases
from qsirecon.utils.transforms import ANAT_SPACE, Xfm

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
    return calls


def _atlas_dataset(root, copies, cohort_in_filename=True):
    """Write a BIDS-Atlas dataset with empty images.

    ``copies`` maps an atlas name to the spaces it is available in.
    BIDS-Atlas allows a cohort to be given by its folder alone, which
    ``cohort_in_filename=False`` writes.
    """
    root.mkdir(parents=True)
    description = {'Name': 'Test atlases', 'BIDSVersion': '1.9.0', 'DatasetType': 'derivative'}
    (root / 'dataset_description.json').write_text(json.dumps(description))
    for atlas, spaces in copies.items():
        for space in spaces:
            template, _, cohort = space.partition('+')
            folder = root / f'tpl-{template}'
            prefix = f'tpl-{template}'
            if cohort:
                folder = folder / f'cohort-{cohort}'
                if cohort_in_filename:
                    prefix += f'_cohort-{cohort}'

            folder.mkdir(parents=True, exist_ok=True)
            stem = f'{prefix}_atlas-{atlas}_dseg'
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
