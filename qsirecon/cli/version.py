# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Version CLI helpers."""

from datetime import datetime
from pathlib import Path

import requests

from .. import __version__

RELEASE_EXPIRY_DAYS = 14
DATE_FMT = '%Y%m%d'
# QSIRecon is not published on PyPI; Docker Hub is where releases are distributed.
TAGS_URL = 'https://hub.docker.com/v2/repositories/pennlinc/qsirecon/tags?page_size=100'
FLAGGED_URL = 'https://raw.githubusercontent.com/PennLINC/qsirecon/main/.versions.json'


def check_latest():
    """Determine the latest released version.

    The latest release is the highest image tag on Docker Hub that parses as a
    final (non-pre-, non-dev-) release. The result is cached in
    ``~/.cache/qsirecon/latest`` for :data:`RELEASE_EXPIRY_DAYS` days.

    Returns
    -------
    latest : :obj:`packaging.version.Version` or None
        The latest release, or None if it could not be determined.
    """
    from packaging.version import InvalidVersion, Version

    latest = None
    date = None
    outdated = None
    cachefile = Path.home() / '.cache' / 'qsirecon' / 'latest'
    try:
        cachefile.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        cachefile = None

    if cachefile and cachefile.exists():
        try:
            latest, date = cachefile.read_text().split('|')
        except Exception:
            pass
        else:
            try:
                latest = Version(latest)
                date = datetime.strptime(date, DATE_FMT)
            except (InvalidVersion, ValueError):
                latest = None
            else:
                if abs((datetime.now() - date).days) > RELEASE_EXPIRY_DAYS:
                    outdated = True

    if latest is None or outdated is True:
        latest = None
        # Nothing about the response (rate limiting, a captive portal, an odd tag name)
        # should be able to stop a run.
        try:
            response = requests.get(url=TAGS_URL, timeout=1.0)
            if response.status_code == 200:
                versions = []
                for tag in response.json()['results']:
                    # Tags like "latest" and "unstable" are not versions.
                    try:
                        versions.append(Version(tag['name']))
                    except (InvalidVersion, KeyError, TypeError):
                        continue
                versions = [v for v in versions if not (v.is_prerelease or v.is_devrelease)]
                if versions:
                    latest = max(versions)
        except Exception:
            latest = None

    if cachefile is not None and latest is not None:
        try:
            cachefile.write_text('|'.join((f'{latest}', datetime.now().strftime(DATE_FMT))))
        except Exception:
            pass

    return latest


def is_flagged():
    """Check whether current version is flagged.

    Flagged versions are listed in the ``.versions.json`` file at the root of the
    repository's ``main`` branch, as a mapping from version string to the reason
    the version was flagged (or ``null`` if no reason is given).

    Returns
    -------
    flagged : :obj:`bool`
        Whether the current version has been flagged.
    reason : :obj:`str` or None
        The reason the version was flagged, if one was given.
    """
    flagged = {}
    # Nothing about the remote file (or whatever is served in its place, e.g., by
    # a captive portal) should be able to stop a run.
    try:
        response = requests.get(url=FLAGGED_URL, timeout=1.0)
        if response.status_code == 200:
            flagged = response.json().get('flagged', {}) or {}
    except Exception:
        flagged = {}

    if isinstance(flagged, dict) and __version__ in flagged:
        return True, flagged[__version__]

    return False, None
