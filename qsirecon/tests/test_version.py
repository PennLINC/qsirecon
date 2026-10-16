"""Test version checks."""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from packaging.version import Version

from qsirecon.cli import version as _version
from qsirecon.cli.version import DATE_FMT, check_latest, is_flagged, requests


def _tags(*names):
    """Shape tag names like the Docker Hub tags API response."""
    return {'count': len(names), 'results': [{'name': name} for name in names]}


TAGS = _tags('latest', '1.0.0', '1.0.1', '1.1.0', '1.1.1rc1', 'unstable')


class MockResponse:
    """Mocks the requests module so that Docker Hub is not actually queried."""

    status_code = 200
    _json = TAGS

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


def test_check_latest1(tmp_path, monkeypatch):
    """Test latest version check."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert str(Path.home()) == str(tmp_path)

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, 'get', mock_get)

    # Initially, cache should not exist
    cachefile = Path.home() / '.cache' / 'qsirecon' / 'latest'
    assert not cachefile.exists()

    # First check actually fetches from Docker Hub
    v = check_latest()
    assert cachefile.exists()
    assert isinstance(v, Version)
    assert v == Version('1.1.0')
    assert cachefile.read_text().split('|') == [str(v), datetime.now().strftime(DATE_FMT)]

    # Second check - test the cache file is read
    cachefile.write_text('|'.join(('1.0.0', cachefile.read_text().split('|')[1])))
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.0.0')

    # Third check - forced outdating of cache
    cachefile.write_text('2.0.0|20180121')
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.1.0')

    # Mock timeouts
    def mock_get(*args, **kwargs):
        raise requests.exceptions.Timeout

    monkeypatch.setattr(requests, 'get', mock_get)

    cachefile.write_text('|'.join(('1.0.0', cachefile.read_text().split('|')[1])))
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.0.0')

    cachefile.write_text('2.0.0|20180121')
    v = check_latest()
    assert v is None

    cachefile.unlink()
    v = check_latest()
    assert v is None


@pytest.mark.parametrize(
    ('result', 'code', 'json'),
    [
        (None, 404, None),
        # Error responses carry a message rather than a list of tags
        (None, 429, {'message': 'too many requests'}),
        (None, 200, {'message': 'too many requests'}),
        (None, 200, []),
        (None, 200, _tags('1.0.0rc1')),
        (None, 200, _tags('26.1.0.dev1')),
        (None, 200, _tags('latest', 'unstable')),
        (None, 200, _tags()),
        (Version('1.1.0'), 200, None),
        (Version('1.0.0'), 200, _tags('1.0.0')),
        # Tags that are not versions are skipped, as are missing names
        (Version('1.0.0'), 200, {'results': [{'name': 'not-a-version'}, {'name': '1.0.0'}, {}]}),
        # Order of the tags in the response does not matter
        (Version('1.1.0'), 200, {'results': TAGS['results'][::-1]}),
    ],
)
def test_check_latest2(tmp_path, monkeypatch, result, code, json):
    """Test latest version check with varying server responses."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert str(Path.home()) == str(tmp_path)

    def mock_get(*args, **kwargs):
        return MockResponse(code=code, json=json)

    monkeypatch.setattr(requests, 'get', mock_get)

    v = check_latest()
    if result is None:
        assert v is None
    else:
        assert isinstance(v, Version)
        assert v == result


def test_check_latest_not_json(tmp_path, monkeypatch):
    """Test latest version check when the response body is not JSON."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))

    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: MockBadResponse())

    assert check_latest() is None


@pytest.mark.parametrize(
    'bad_cache',
    [
        '3laj#r???d|3akajdf#',
        '2.0.0|3akajdf#',
        '|'.join(('2.0.0', datetime.now().strftime(DATE_FMT), '')),
        '',
    ],
)
def test_check_latest3(tmp_path, monkeypatch, bad_cache):
    """Test latest version check when the cache file is corrupted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert str(Path.home()) == str(tmp_path)

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, 'get', mock_get)

    # Initially, cache should not exist
    cachefile = Path.home() / '.cache' / 'qsirecon' / 'latest'
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    assert not cachefile.exists()

    cachefile.write_text(bad_cache)
    v = check_latest()
    assert isinstance(v, Version)
    assert v == Version('1.1.0')


@pytest.mark.skipif(not hasattr(os, 'geteuid'), reason='POSIX permissions required')
def test_readonly(tmp_path, monkeypatch):
    """Test behavior when $HOME/.cache/qsirecon/latest can't be written out."""
    home_path = Path('/home/readonly') if os.getenv('TEST_READONLY_FILESYSTEM') else tmp_path
    monkeypatch.setenv('HOME', str(home_path))
    cachedir = home_path / '.cache'

    if os.getenv('TEST_READONLY_FILESYSTEM') is None:
        if os.geteuid() == 0:
            pytest.skip('Cannot mock being unable to create directories as root')
        cachedir.mkdir(mode=0o555, exist_ok=True)

    # Make sure creating the folder will raise the exception.
    with pytest.raises(OSError, match=r'Read-only file system|Permission denied'):
        (cachedir / 'qsirecon').mkdir(parents=True)

    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: MockResponse())

    # Should not raise
    assert check_latest() == Version('1.1.0')


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
