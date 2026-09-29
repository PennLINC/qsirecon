# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Version CLI helpers."""

from datetime import datetime
from pathlib import Path

import requests

from .. import __version__

RELEASE_EXPIRY_DAYS = 14
DATE_FMT = '%Y%m%d'
FLAGGED_URL = 'https://raw.githubusercontent.com/PennLINC/qsirecon/main/.versions.json'


def check_latest():
    """Determine whether this is the latest version."""
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
        try:
            response = requests.get(url='https://pypi.org/pypi/qsirecon/json', timeout=1.0)
        except Exception:
            response = None

        if response and response.status_code == 200:
            versions = [Version(rel) for rel in response.json()['releases'].keys()]
            versions = [rel for rel in versions if not rel.is_prerelease]
            if versions:
                latest = sorted(versions)[-1]
        else:
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
