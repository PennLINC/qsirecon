import sys
from pathlib import Path

# Repo root on path so ``import qsirecon`` works without pip install (CircleCI VM).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from qsirecon.tests.integration_test_data import download_test_data

if __name__ == '__main__':
    data_dir = sys.argv[1]
    dset = sys.argv[2]
    download_test_data(dset, data_dir)
