"""Download integration test archives.

Uses only the standard library so ``.circleci/get_data.py`` can run on CI VMs
without installing qsirecon or test dependencies.
"""

import lzma
import os
import tarfile
import urllib.request
from gzip import GzipFile
from io import BytesIO

_INTEGRATION_TEST_DATA_URLS = {
    'multishell_output': (
        'https://upenn.box.com/shared/static/q066t33ojqk1phljr6064ilcrrqip87q.gz'
    ),
    'singleshell_output': (
        'https://upenn.box.com/shared/static/rc4oh9tqf20f2xu2p8587q44vwr1i7uf.gz'
    ),
    'hsvs_data': ('https://upenn.box.com/shared/static/kic35jscwqhtezi6t398k4nfuo09ppty.xz'),
    'multises_pre1_output': (
        'https://upenn.box.com/shared/static/g6hb4ylraejqn8sjz2xodj6fog4lp6sb.xz'
    ),
    'multises_post1_output': (
        'https://upenn.box.com/shared/static/ipqhy6a9p0pl7q1tw4zejj47mro4dtfh.xz'
    ),
    'custom_atlases': (
        'https://upenn.box.com/shared/static/2x5jqj7he20lminc3v4jtwhyiq7pyr8i.gz'
    ),
}


def _default_test_data_parent() -> str:
    return os.path.join(os.path.dirname(__file__), 'data')


def download_test_data(dset, data_dir=None, info=print):
    """Download a named integration-test dataset into ``data_dir``."""
    URLS = _INTEGRATION_TEST_DATA_URLS
    if dset == '*':
        for k in URLS:
            download_test_data(k, data_dir=data_dir, info=info)
        return

    if dset not in URLS:
        raise ValueError(f'dset ({dset}) must be one of: {", ".join(URLS.keys())}')

    if not data_dir:
        data_dir = os.path.join(_default_test_data_parent(), 'test_data')

    out_dir = os.path.join(data_dir, dset)

    if os.path.isdir(out_dir):
        info(
            f'Dataset {dset} already exists. '
            'If you need to re-download the data, please delete the folder.'
        )
        return out_dir

    info(f'Downloading {dset} to {out_dir}')

    os.makedirs(out_dir, exist_ok=True)
    url = URLS[dset]
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()

    if url.endswith('.xz'):
        with lzma.open(BytesIO(raw)) as f:
            with tarfile.open(fileobj=f) as t:
                t.extractall(out_dir)
    elif url.endswith('.gz'):
        with tarfile.open(fileobj=GzipFile(fileobj=BytesIO(raw))) as t:
            t.extractall(out_dir)
    else:
        raise ValueError(f'Unknown file type for {dset} ({url})')

    return out_dir
