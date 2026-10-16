"""Test parser."""

import pytest
from packaging.version import Version

from qsirecon import config
from qsirecon.cli import version as _version
from qsirecon.cli.parser import _build_parser


@pytest.mark.parametrize(('current', 'latest'), [('1.0.0', '1.3.2'), ('1.3.2', '1.3.2')])
def test_get_parser_update(monkeypatch, capsys, current, latest):
    """Make sure the out-of-date banner is shown."""
    expectation = Version(current) < Version(latest)

    def _mock_check_latest(*args, **kwargs):
        return Version(latest)

    monkeypatch.setattr(config.environment, 'version', current)
    monkeypatch.setattr(_version, 'check_latest', _mock_check_latest)
    monkeypatch.setattr(_version, 'is_flagged', lambda *args, **kwargs: (False, None))

    _build_parser()
    captured = capsys.readouterr().err

    msg = f"""\
You are using QSIRecon-{current}, and a newer version of QSIRecon is available: {latest}.
Please check out our documentation about how and when to upgrade:
https://qsirecon.readthedocs.io/en/latest/installation.html#upgrading"""

    assert (msg in captured) is expectation


@pytest.mark.parametrize('flagged', [(True, None), (True, 'random reason'), (False, None)])
def test_get_parser_flagged(monkeypatch, capsys, flagged):
    """Make sure the flagged-version banner is shown."""

    def _mock_is_flagged(*args, **kwargs):
        return flagged

    monkeypatch.setattr(_version, 'check_latest', lambda *args, **kwargs: None)
    monkeypatch.setattr(_version, 'is_flagged', _mock_is_flagged)

    _build_parser()
    captured = capsys.readouterr().err

    assert ('FLAGGED' in captured) is flagged[0]
    if flagged[0]:
        assert f'reason: {flagged[1] or "unknown"}' in captured
