"""Test parser."""

import pytest

from qsirecon.cli import version as _version
from qsirecon.cli.parser import _build_parser


@pytest.mark.parametrize('flagged', [(True, None), (True, 'random reason'), (False, None)])
def test_get_parser_flagged(monkeypatch, capsys, flagged):
    """Make sure the flagged-version banner is shown."""

    def _mock_is_flagged(*args, **kwargs):
        return flagged

    monkeypatch.setattr(_version, 'is_flagged', _mock_is_flagged)

    _build_parser()
    captured = capsys.readouterr().err

    assert ('FLAGGED' in captured) is flagged[0]
    if flagged[0]:
        assert f'reason: {flagged[1] or "unknown"}' in captured
