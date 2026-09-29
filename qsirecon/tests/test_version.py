"""Test version checks."""

import json
from pathlib import Path

import pytest
from packaging.version import Version

from qsirecon.cli import version as _version
from qsirecon.cli.version import is_flagged, requests


class MockResponse:
    """Mocks the requests module so that GitHub is not actually queried."""

    status_code = 200
    _json = {'flagged': {}}

    def __init__(self, code=200, json=None):
        """Allow setting different response codes."""
        self.status_code = code
        if json is not None:
            self._json = json

    def json(self):
        """Redefine the response object."""
        return self._json


class MockBadResponse(MockResponse):
    """Mocks a response whose body is not JSON (e.g., from a captive portal)."""

    def json(self):
        """Fail to decode the body."""
        raise requests.exceptions.JSONDecodeError('Expecting value', '<html>', 0)


@pytest.mark.parametrize(
    ('result', 'version', 'code', 'json'),
    [
        (False, '1.2.1', 200, {'flagged': {'1.0.0': None}}),
        (True, '1.2.1', 200, {'flagged': {'1.2.1': None}}),
        (True, '1.2.1', 200, {'flagged': {'1.2.1': 'FATAL Bug!'}}),
        (False, '1.2.1', 404, {'flagged': {'1.0.0': None}}),
        (False, '1.2.1', 404, {'flagged': {'1.2.1': 'FATAL Bug!'}}),
        (False, '1.2.1', 200, {'flagged': []}),
        (False, '1.2.1', 200, {'flagged': ['1.2.1']}),
        (False, '1.2.1', 200, {}),
        (False, '1.2.1', 200, ['1.2.1']),
    ],
)
def test_is_flagged(monkeypatch, result, version, code, json):
    """Test that the flagged-versions check is correct."""
    monkeypatch.setattr(_version, '__version__', version)

    def mock_get(*args, **kwargs):
        return MockResponse(code=code, json=json)

    monkeypatch.setattr(requests, 'get', mock_get)

    val, reason = is_flagged()
    assert val is result

    test_reason = None
    if val:
        test_reason = json.get('flagged', {}).get(version, None)

    if test_reason is not None:
        assert reason == test_reason
    else:
        assert reason is None


@pytest.mark.parametrize(
    'mock_get',
    [
        pytest.param(lambda *args, **kwargs: MockBadResponse(), id='not-json'),
        pytest.param(
            lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.Timeout),
            id='timeout',
        ),
        pytest.param(
            lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ConnectionError),
            id='offline',
        ),
    ],
)
def test_is_flagged_unreachable(monkeypatch, mock_get):
    """Test that failing to retrieve the flagged versions never raises."""
    monkeypatch.setattr(_version, '__version__', '1.2.1')
    monkeypatch.setattr(requests, 'get', mock_get)

    assert is_flagged() == (False, None)


def test_versions_file():
    """Check that the list of flagged versions in the repository is well-formed."""
    versions_file = Path(__file__).parents[2] / '.versions.json'
    if not versions_file.exists():
        pytest.skip('Not running from a source checkout')

    flagged = json.loads(versions_file.read_text())['flagged']
    assert isinstance(flagged, dict)
    for version, reason in flagged.items():
        # The check is an exact string match against qsirecon.__version__,
        # so the key must be written the way the version is normalized.
        assert str(Version(version)) == version
        assert reason is None or isinstance(reason, str)
