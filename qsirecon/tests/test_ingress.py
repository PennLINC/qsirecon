"""QSIPrepDWIIngress resolves the dwiref under both QSIPrep naming schemes.

QSIPrep >= 27 names the b=0 reference after the series it accompanies
(``..._desc-preproc_dwiref.nii.gz``), so it cannot collide with the subject-level
dwiref template. Earlier versions wrote it without the ``desc``. QSIRecon has to
read both: the rename is not a reason to stop ingesting 26.x derivatives.
"""

import pytest

from qsirecon.interfaces.ingress import QSIPrepDWIIngress

PREFIX = 'sub-01_space-ACPC_desc-preproc'
NEW_NAME = f'{PREFIX}_dwiref.nii.gz'
OLD_NAME = 'sub-01_space-ACPC_dwiref.nii.gz'


def _derivatives(tmp_path, reference_names):
    """A minimal QSIPrep-style output tree; returns the preprocessed series."""
    dwi_dir = tmp_path / 'sub-01' / 'dwi'
    dwi_dir.mkdir(parents=True)

    series = dwi_dir / f'{PREFIX}_dwi.nii.gz'
    series.write_text('')
    for ext in ('.bval', '.bvec'):
        (dwi_dir / f'{PREFIX}_dwi{ext}').write_text('')
    for name in reference_names:
        (dwi_dir / name).write_text('')
    return series


@pytest.mark.parametrize(
    ('present', 'expected'),
    [
        pytest.param([NEW_NAME], NEW_NAME, id='new-only'),
        pytest.param([OLD_NAME], OLD_NAME, id='old-only'),
        # Precedence: a tree holding both must resolve to the new name. This is
        # what forces real conditional logic -- _get_if_exists mutates _results
        # and returns nothing, so it cannot be chained as an expression.
        pytest.param([NEW_NAME, OLD_NAME], NEW_NAME, id='both-new-wins'),
    ],
)
def test_dwiref_resolution(tmp_path, present, expected):
    series = _derivatives(tmp_path, present)
    result = QSIPrepDWIIngress(dwi_file=str(series)).run()
    assert result.outputs.dwi_ref.endswith(expected)


def test_dwiref_is_absent_when_no_reference_exists(tmp_path):
    """A missing reference is tolerated, not an error."""
    from nipype.interfaces.base import isdefined

    series = _derivatives(tmp_path, [])
    result = QSIPrepDWIIngress(dwi_file=str(series)).run()
    assert not isdefined(result.outputs.dwi_ref)
