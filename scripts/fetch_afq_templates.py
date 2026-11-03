#!/usr/bin/env python
"""Pre-fetch the pyAFQ bundle templates that QSIRecon's pyAFQ workflows use.

pyAFQ downloads its templates from figshare into ``$AFQ_HOME`` (``~/AFQ_data`` by default)
the first time a bundle dictionary needs them. Caching them in the Docker image keeps that
network access out of every run and out of CI.

The files are downloaded with a plain ``urllib`` request rather than through DIPY's fetcher.
DIPY 1.12 sends browser-like headers to figshare, which answers them with a bot-check page
instead of the file, and pyAFQ does not verify checksums, so the page would be cached as a
corrupt ``.nii.gz``. Each download is checked to be gzip data before it is written.
"""

import argparse
import logging
import os
import os.path as op
import time
import urllib.parse
import urllib.request

import dipy.data.fetcher as dipy_fetcher
import nibabel as nb

GZIP_MAGIC = b'\x1f\x8b'
N_ATTEMPTS = 5
LOGGER = logging.getLogger('fetch_afq_templates')


def _download(fname, url, **kwargs):
    """Download ``url`` to ``fname`` with a plain client, retrying until the data is gzip.

    Replaces ``dipy.data.fetcher._get_file_data`` while the pyAFQ fetchers run, so pyAFQ's
    own file lists decide what is downloaded. Extra keyword arguments from DIPY are ignored.
    """
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError(f'Refusing to download {url}: only https URLs are expected')

    for attempt in range(1, N_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                data = response.read()
        except OSError as err:
            LOGGER.warning(f'Attempt {attempt} for {url} failed: {err}')
        else:
            if data[:2] == GZIP_MAGIC:
                with open(fname, 'wb') as fobj:
                    fobj.write(data)

                return

            LOGGER.warning(f'Attempt {attempt} for {url} did not return gzip data')

        time.sleep(2**attempt)

    raise RuntimeError(f'Could not download {url} after {N_ATTEMPTS} attempts')


def fetch_all():
    """Fetch the templates behind pyAFQ's default, SLF and callosal bundle dictionaries."""
    import AFQ.data.fetch as afd

    if not hasattr(dipy_fetcher, '_get_file_data'):
        raise RuntimeError('dipy.data.fetcher has no _get_file_data to replace')

    dipy_fetcher._get_file_data = _download
    folders = {}
    for fetcher in (
        afd.fetch_templates,
        afd.fetch_callosum_templates,
        afd.fetch_or_templates,
        afd.fetch_slf_templates,
    ):
        LOGGER.info(f'Running {fetcher.__name__}')
        files, folder = fetcher()
        folders[folder] = sorted(files)

    return folders


def check_all(folders):
    """Make sure every expected file is present and is a readable NIfTI image."""
    n_files = 0
    for folder, fnames in folders.items():
        for fname in fnames:
            path = op.join(folder, fname)
            with open(path, 'rb') as fobj:
                if fobj.read(2) != GZIP_MAGIC:
                    raise RuntimeError(f'{path} is not a gzip file')

            if nb.load(path).ndim < 3:
                raise RuntimeError(f'{path} is not a 3D image')

            n_files += 1

        LOGGER.info(f'{folder}: {len(fnames)} files')

    LOGGER.info(f'{n_files} template files are in place')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-cache the pyAFQ templates that QSIRecon needs',
    )
    parser.add_argument(
        '--afq-dir',
        type=os.path.abspath,
        help=(
            'Directory to save templates in. '
            'If not provided, templates will be saved to `$AFQ_HOME` or `${HOME}/AFQ_data`.'
        ),
    )
    opts = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    if opts.afq_dir is not None:
        os.environ['AFQ_HOME'] = opts.afq_dir

    check_all(fetch_all())
