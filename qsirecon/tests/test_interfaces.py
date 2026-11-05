"""Command-line interface tests."""

import logging
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.reconst.force import FORCEModel

from qsirecon.data import load as load_data
from qsirecon.interfaces.amico import _use_image_amico_matrices
from qsirecon.interfaces.dipy import FORCEReconstruction
from qsirecon.interfaces.gradients import GradientSelect, _classify_shell_scheme, _find_shells
from qsirecon.interfaces.utils import AtlasLUTs
from qsirecon.tests.utils import download_test_data, get_test_data_path


def _write_labels_file(tmpdir, rows):
    """Write a minimal atlas labels TSV with index and name columns."""
    labels_file = Path(tmpdir) / 'atlas_dseg.tsv'
    with open(labels_file, 'w') as fo:
        fo.write('index\tname\n')
        fo.writelines(f'{index}\t{name}\n' for index, name in rows)

    return labels_file


def test_atlas_luts(tmp_path_factory):
    """AtlasLUTs writes the original indices and sequential MRtrix indices."""
    tmpdir = tmp_path_factory.mktemp('test_atlas_luts')
    labels_file = _write_labels_file(tmpdir, [(5, 'Left Thalamus'), (12, 'Right Thalamus')])

    interface = AtlasLUTs(atlas_labels_file=str(labels_file))
    results = interface.run(cwd=str(tmpdir))

    with open(results.outputs.orig_lut) as fo:
        assert fo.read() == '5\tLeft-Thalamus\n12\tRight-Thalamus\n'

    with open(results.outputs.mrtrix_lut) as fo:
        assert fo.read() == '1\tLeft-Thalamus\n2\tRight-Thalamus\n'


def test_atlas_luts_drops_zero_index(tmp_path_factory):
    """AtlasLUTs drops the background (0) index from both LUTs."""
    tmpdir = tmp_path_factory.mktemp('test_atlas_luts_drops_zero_index')
    labels_file = _write_labels_file(tmpdir, [(0, 'background'), (5, 'Left Thalamus')])

    interface = AtlasLUTs(atlas_labels_file=str(labels_file))
    results = interface.run(cwd=str(tmpdir))

    with open(results.outputs.orig_lut) as fo:
        assert fo.read() == '5\tLeft-Thalamus\n'

    with open(results.outputs.mrtrix_lut) as fo:
        assert fo.read() == '1\tLeft-Thalamus\n'


def test_shell_selection(data_dir, tmp_path_factory):
    """Run reconstruction workflow tests."""
    dwi_prefix = 'sub-ABCD_acq-10per000_space-T1w_desc-preproc_dwi'
    tmpdir = tmp_path_factory.mktemp('test_shell_selection')
    dataset_dir = Path(download_test_data('multishell_output', data_dir))
    dataset_dir = Path(dataset_dir) / 'multishell_output' / 'qsiprep'
    data_stem = str(dataset_dir / 'sub-ABCD' / 'dwi' / dwi_prefix)

    # Test actual data (including a nifti)
    grad_select = GradientSelect(
        dwi_file=data_stem + '.nii.gz',
        bval_file=data_stem + '.bval',
        bvec_file=data_stem + '.bvec',
        b_file=data_stem + '.b',
        bval_distance_cutoff=100,
        expected_n_input_shells=5,
        requested_shells=[0, 'lowest', 'highest'],
    )
    grad_select.run(cwd=tmpdir)
    correct_n = 73
    sel_nii = nb.load(tmpdir / (dwi_prefix + '_selected.nii.gz'))
    assert sel_nii.shape[3] == correct_n
    sel_bval = np.loadtxt(tmpdir / (dwi_prefix + '_selected.bval'))
    assert sel_bval.shape == (correct_n,)
    sel_bvec = np.loadtxt(tmpdir / (dwi_prefix + '_selected.bvec'))
    assert sel_bvec.shape == (3, correct_n)
    sel_b = np.loadtxt(tmpdir / (dwi_prefix + '_selected.b'))
    assert sel_b.shape[0] == correct_n
    # There is no btable from this dataset because it was created
    # before those were written in the outputs.
    assert not Path(tmpdir / (dwi_prefix + '_selected.txt')).exists()


def test_real_shells():
    """Test the _find_shells function."""
    # Check some other sequences we might run into
    # ABCD has 4 shells + b=0s
    abcd_bval = np.loadtxt(load_data('schemes/ABCD.bval'))
    abcd_df = _find_shells(abcd_bval, 100)
    assert abcd_df['shell_num'].max() == 5.0

    # HCP has 3 shells + b=0s
    hcp_bval = np.loadtxt(load_data('schemes/HCP.bval'))
    hcp_df = _find_shells(hcp_bval, 100)
    assert hcp_df['shell_num'].max() == 4.0

    # DSIQ5 should raise an exception
    dsi_bval = np.loadtxt(load_data('schemes/DSIQ5.bval'))
    with pytest.raises(Exception, match=r'Too many possible shells detected\.'):
        _find_shells(dsi_bval, 100)

    # Some other assorted test schemes that should fail
    nonshelled_schemes = ['HASC55-1', 'HASC55-2', 'HASC92', 'RAND57']
    for scheme in nonshelled_schemes:
        bval_file = Path(get_test_data_path()) / f'{scheme}.bval'
        test_bvals = np.loadtxt(bval_file)

        with pytest.raises(Exception, match=r'Too many possible shells detected\.'):
            _find_shells(test_bvals, 100)

    # DSIQ7 actually _passes_ the mrtrix test, but fails silhouette
    bval_file = Path(get_test_data_path()) / 'Q7.bval'
    test_bvals = np.loadtxt(bval_file)

    with pytest.raises(Exception, match=r'Silhouette score is low\. Is this is a DSI scheme\?'):
        _find_shells(test_bvals, 100)


def test_classify_shell_scheme(tmp_path_factory):
    """Test the _classify_shell_scheme function."""
    tmpdir = tmp_path_factory.mktemp('test_classify_shell_scheme')
    # Check some other sequences we might run into
    # ABCD has 4 shells + b=0s
    abcd_bval = load_data('schemes/ABCD.bval')
    scheme = _classify_shell_scheme(abcd_bval, 5)
    assert scheme == 'multishell'

    # HCP has 3 shells + b=0s
    hcp_bval = load_data('schemes/HCP.bval')
    scheme = _classify_shell_scheme(hcp_bval, 5)
    assert scheme == 'multishell'

    # DSIQ5 should raise an exception
    dsi_bval = load_data('schemes/DSIQ5.bval')
    scheme = _classify_shell_scheme(dsi_bval, 5)
    assert scheme == 'non-shelled'

    # Some other assorted test schemes that should fail
    nonshelled_schemes = ['HASC55-1', 'HASC55-2', 'HASC92', 'RAND57']
    for scheme_name in nonshelled_schemes:
        bval_file = Path(get_test_data_path()) / f'{scheme_name}.bval'
        scheme = _classify_shell_scheme(bval_file, 5)
        assert scheme == 'non-shelled'

    oasis = [
        0,
        50,
        350,
        600,
        900,
        1150,
        100,
        400,
        650,
        950,
        150,
        450,
        700,
        1000,
        1300,
        200,
        500,
        800,
        1050,
        1350,
        300,
        850,
        1100,
        1400,
    ]
    oasis_bval_file = str(tmpdir / 'oasis.bval')
    np.savetxt(oasis_bval_file, np.array(oasis))
    scheme = _classify_shell_scheme(oasis_bval_file, 5)
    assert scheme == 'unknown'


def _get_force_simulations(tmp_path, **inputs):
    """Run FORCEReconstruction._get_simulations.

    Returns the model, the working directory, and the SimulationsFile the interface would
    record in its metadata.
    """
    rng = np.random.default_rng(0)
    bvecs = rng.normal(size=(61, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs[0] = 0
    bvals = np.r_[0, [1000] * 30, [2000] * 30]
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)

    dwi_file = tmp_path / 'sub-01_dwi.nii.gz'
    dwi_file.touch()
    work_dir = tmp_path / 'work'
    work_dir.mkdir(exist_ok=True)

    interface = FORCEReconstruction(dwi_file=str(dwi_file), num_simulations=200, **inputs)
    model = FORCEModel(gtab, verbose=False)
    interface._get_simulations(model, SimpleNamespace(cwd=str(work_dir)))
    metadata = interface._list_outputs()['fa_metadata']
    return model, work_dir, metadata['Model']['Parameters']['SimulationsFile']


def test_force_simulations_default_to_working_directory(tmp_path, monkeypatch):
    """Without DIPY_HOME, the library goes to the working directory.

    DIPY's own cache would then be ``~/.dipy/force_simulations``, which is read-only when
    the home directory is inside an Apptainer image. The home directory is placed
    underneath a regular file here, so any attempt to write to it raises an error.
    """
    blocker = tmp_path / 'blocker'
    blocker.write_text('')
    monkeypatch.setenv('HOME', str(blocker / 'home'))
    monkeypatch.delenv('DIPY_HOME', raising=False)

    model, work_dir, recorded_file = _get_force_simulations(tmp_path)

    assert [path.name for path in work_dir.iterdir()] == ['sub-01_dwi_force_simulations.npz']
    assert model.simulations is not None
    assert recorded_file == 'n/a'


def test_force_simulations_cached_in_dipy_home(tmp_path, monkeypatch):
    """DIPY_HOME stores the library and reuses it for matching runs."""
    dipy_home = tmp_path / 'dipy_home'
    monkeypatch.setenv('DIPY_HOME', str(dipy_home))
    cache_dir = dipy_home / 'force_simulations'

    first_model, work_dir, recorded_file = _get_force_simulations(tmp_path)
    cached_files = list(cache_dir.iterdir())
    assert len(cached_files) == 1
    assert cached_files[0].name.startswith('qsirecon_')
    assert not list(work_dir.iterdir())
    assert recorded_file == str(cached_files[0])

    # The same gradient table and parameters load the stored library instead of
    # generating a new one.
    second_model, _, recorded_file = _get_force_simulations(tmp_path)
    assert list(cache_dir.iterdir()) == cached_files
    assert recorded_file == str(cached_files[0])
    for name, values in first_model.simulations.items():
        np.testing.assert_array_equal(second_model.simulations[name], values)

    # A parameter that DIPY's own cache registry ignores still gets its own library.
    _get_force_simulations(tmp_path, compute_dki=True)
    assert len(list(cache_dir.iterdir())) == 2


@pytest.mark.parametrize('use_dipy_home', [False, True])
def test_force_scratch_files_in_working_directory(tmp_path, monkeypatch, use_dipy_home):
    """DIPY's scratch arrays go to the working directory, not to /tmp.

    Under ``apptainer --containall``, /tmp is a small tmpfs, and filling it from a
    memory-mapped file kills the process with SIGBUS.
    """
    import dipy.sims.force

    if use_dipy_home:
        monkeypatch.setenv('DIPY_HOME', str(tmp_path / 'dipy_home'))
    else:
        monkeypatch.delenv('DIPY_HOME', raising=False)

    scratch_dirs = set()
    create_memmap = dipy.sims.force._create_memmap

    def record_memmap_dir(output_dir, *args, **kwargs):
        scratch_dirs.add(Path(output_dir))
        return create_memmap(output_dir, *args, **kwargs)

    monkeypatch.setattr(dipy.sims.force, '_create_memmap', record_memmap_dir)
    tempdir = tempfile.tempdir
    _, work_dir, _ = _get_force_simulations(tmp_path)

    assert len(scratch_dirs) == 1
    assert next(iter(scratch_dirs)).parent == work_dir
    assert tempfile.tempdir == tempdir
    # DIPY removes its scratch directory once the library is built.
    expected = [] if use_dipy_home else ['sub-01_dwi_force_simulations.npz']
    assert [path.name for path in work_dir.iterdir()] == expected


def test_force_simulations_concurrent_miss(tmp_path, monkeypatch):
    """A run that loses the race to store a library fits against the stored one."""
    dipy_home = tmp_path / 'dipy_home'
    monkeypatch.setenv('DIPY_HOME', str(dipy_home))
    cache_dir = dipy_home / 'force_simulations'

    # Take the library that another run generates out of the cache for now.
    other_model, _, _ = _get_force_simulations(tmp_path)
    cached_file = next(cache_dir.iterdir())
    other_file = tmp_path / 'other_run.npz'
    shutil.move(cached_file, other_file)

    # The other run stores its library while this one is still generating.
    generate = FORCEModel.generate

    def generate_then_lose_race(self, *args, **kwargs):
        result = generate(self, *args, **kwargs)
        shutil.copyfile(other_file, cached_file)
        return result

    monkeypatch.setattr(FORCEModel, 'generate', generate_then_lose_race)
    model, _, recorded_file = _get_force_simulations(tmp_path)

    assert list(cache_dir.iterdir()) == [cached_file]
    assert recorded_file == str(cached_file)
    for name, values in other_model.simulations.items():
        np.testing.assert_array_equal(model.simulations[name], values)


@pytest.fixture
def amico_lut():
    """Import ``amico.lut`` without leaving AMICO's logger class installed.

    Importing AMICO makes dicelib install its own logger class, whose ``warning`` takes
    ``stacklevel`` as its second positional argument. Any logger created after that breaks
    on ``logger.warning('%s', value)``, which is what matplotlib does while it is imported.
    So AMICO is imported here rather than at the top of the module, and the logger class
    is put back afterwards.
    """
    logger_class = logging.getLoggerClass()
    import amico

    logging.setLoggerClass(logger_class)
    return amico.lut


def test_amico_reads_matrices_from_image(tmp_path, monkeypatch, amico_lut):
    """AMICO loads its rotation matrices from QSIRECON_AMICO_HOME, whatever DIPY_HOME is.

    This guards the assumption that AMICO honors ``amico.lut.dipy_home``. In the Docker
    image, ``~/.dipy`` is the same directory as ``QSIRECON_AMICO_HOME``, so NODDI would
    keep working there even if the redirection silently stopped taking effect.
    """
    # Stand in for the image: generate real matrices in a directory of their own.
    amico_home = tmp_path / 'amico_home'
    monkeypatch.setattr(amico_lut, 'dipy_home', str(amico_home))
    amico_lut.precompute_rotation_matrices(12, 500)
    assert [path.name for path in amico_home.iterdir()] == [
        'AMICO_aux_matrices_lmax=12_ndirs=500.pickle'
    ]

    # AMICO starts out looking in a DIPY_HOME that does not have them.
    dipy_home = tmp_path / 'dipy_home'
    monkeypatch.setattr(amico_lut, 'dipy_home', str(dipy_home))
    monkeypatch.setenv('QSIRECON_AMICO_HOME', str(amico_home))

    _use_image_amico_matrices()
    assert amico_lut.load_precomputed_rotation_matrices(12, 500)['ndirs'] == 500
    assert not dipy_home.exists()


def test_amico_left_alone_outside_image(tmp_path, monkeypatch, amico_lut):
    """Without QSIRECON_AMICO_HOME, AMICO keeps looking in DIPY's home directory."""
    dipy_home = tmp_path / 'dipy_home'
    monkeypatch.setattr(amico_lut, 'dipy_home', str(dipy_home))
    monkeypatch.delenv('QSIRECON_AMICO_HOME', raising=False)

    _use_image_amico_matrices()
    assert amico_lut.dipy_home == str(dipy_home)
