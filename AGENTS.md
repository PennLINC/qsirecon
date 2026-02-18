# AGENTS.md -- QSIRecon

This file provides instructions for AI coding agents and human maintainers working on **QSIRecon**, a BIDS App for reconstructing and postprocessing q-space (diffusion) MRI images.

---

## Shared Instructions (All PennLINC BIDS Apps)

The following conventions apply equally to **qsiprep**, **qsirecon**, **xcp_d**, and **aslprep**. All four are PennLINC BIDS Apps built on the NiPreps stack.

### Ecosystem Context

- These projects belong to the [NiPreps](https://www.nipreps.org/) ecosystem and follow its community guidelines.
- Core dependencies include **nipype** (workflow engine), **niworkflows** (reusable workflow components), **nireports** (visual reports), **pybids** (BIDS dataset querying), and **nibabel** (neuroimaging I/O).
- All four apps are containerized via Docker and distributed on Docker Hub under the `pennlinc/` namespace.
- Contributions follow the [NiPreps contributing guidelines](https://www.nipreps.org/community/CONTRIBUTING/).

### Architecture Overview

Every PennLINC BIDS App follows this execution flow:

```
CLI (parser.py / run.py)
  -> config singleton (config.py, serialized as ToML)
    -> workflow graph construction (workflows/*.py)
      -> Nipype interfaces (interfaces/*.py)
        -> BIDS-compliant derivative outputs
```

- **CLI** (`<pkg>/cli/`): `parser.py` defines argparse arguments; `run.py` is the entry point; `workflow.py` builds the execution graph; `version.py` handles `--version`.
- **Config** (`<pkg>/config.py`): A singleton module with class-based sections (`environment`, `execution`, `workflow`, `nipype`, `seeds`). Config is serialized to ToML and passed across processes via the filesystem. Access settings as `config.section.setting`.
- **Workflows** (`<pkg>/workflows/`): Built using `nipype.pipeline.engine` (`pe.Workflow`, `pe.Node`, `pe.MapNode`). Use `LiterateWorkflow` from `niworkflows.engine.workflows` for auto-documentation. Every workflow factory function must be named `init_<descriptive_name>_wf`.
- **Interfaces** (`<pkg>/interfaces/`): Custom Nipype interfaces wrapping external tools or Python logic. Follow standard Nipype patterns: define `_InputSpec` / `_OutputSpec` with `BaseInterfaceInputSpec` / `TraitedSpec`, implement `_run_interface()`.
- **Utilities** (`<pkg>/utils/`): Shared helper functions. BIDS-specific helpers live in `utils/bids.py`.
- **Reports** (`<pkg>/reports/`): HTML report generation using nireports.
- **Data** (`<pkg>/data/`): Static package data (config files, templates, atlases). Accessed via `importlib.resources` or the `acres` package.
- **Tests** (`<pkg>/tests/`): Pytest-based. Unit tests run without external data. Integration tests are gated behind pytest markers and are skipped by default.

### Workflow Authoring Rules

1. Every workflow factory function must be named `init_<name>_wf` and return a `Workflow` object.
2. Use `LiterateWorkflow` (from `niworkflows.engine.workflows`) to enable automatic workflow graph documentation.
3. Define `inputnode` and `outputnode` as `niu.IdentityInterface` nodes to declare the workflow's external API.
4. Connect nodes using `workflow.connect([(source, dest, [('out_field', 'in_field')])])` syntax.
5. Add `# fmt:skip` after multi-line `workflow.connect()` calls to prevent the formatter from reformatting them.
6. Include a docstring with `Workflow Graph` and `.. workflow::` Sphinx directive for auto-generated documentation.
7. Use `config` module values (not function parameters) for global settings inside workflow builders.

### Interface Conventions

1. Input/output specs use Nipype traits (`File`, `traits.Bool`, `traits.Int`, etc.).
2. `mandatory = True` for required inputs; provide `desc=` for all traits.
3. Implement `_run_interface(self, runtime)` -- never `run()`.
4. Return `runtime` from `_run_interface`.
5. Set outputs via `self._results['field'] = value`.

### Config Module Usage

```python
from <pkg> import config

# Read a setting
work_dir = config.execution.work_dir

# Serialize to disk
config.to_filename(path)

# Load from disk (in a subprocess)
config.load(path)
```

The config module is the single source of truth for runtime parameters. Never pass global settings as function arguments when they are available via config.

### Testing Conventions

- **Unit tests**: Files named `test_*.py` in `<pkg>/tests/`. Must not require external neuroimaging data or network access.
- **Integration tests**: Decorated with `@pytest.mark.<marker_name>`. Excluded by default via `addopts` in `pyproject.toml`. Require Docker or pre-downloaded test datasets.
- **Fixtures**: Defined in `conftest.py`. Common fixtures include `data_dir`, `working_dir`, `output_dir`, and `datasets`.
- **Coverage**: Configured in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`.

### Documentation

- Built with Sphinx using `sphinx_rtd_theme`.
- Source files in `docs/`.
- Workflow graphs are auto-rendered via `.. workflow::` directives that call `init_*_wf` functions.
- API docs generated via `sphinxcontrib-apidoc`.
- Bibliography managed with `sphinxcontrib-bibtex` and `boilerplate.bib`.

### Docker

- Each app has a custom base image: `pennlinc/<pkg>_build:<version>`.
- The `Dockerfile` installs the app via `pip install` into the base image.
- Entrypoint is the CLI command (e.g., `/opt/conda/envs/<pkg>/bin/<pkg>`).
- Labels follow the `org.label-schema` convention.

### Release Process

- Versions are derived from git tags via `hatch-vcs` (VCS-based versioning).
- GitHub Releases use auto-generated changelogs configured in `.github/release.yml`.
- Release categories: Breaking Changes, New Features, Deprecations, Bug Fixes, Other.
- Docker images are built and pushed via CI on tagged releases.

### Code Style

- **Formatter**: `ruff format` (target: all four repos).
- **Linter**: `ruff check` with an extended rule set (F, E, W, I, UP, YTT, S, BLE, B, A, C4, DTZ, T10, EXE, FA, ISC, ICN, PT, Q).
- **Import sorting**: Handled by ruff's `I` rule (isort-compatible).
- **Pre-commit**: Uses `ruff-pre-commit` hooks for both linting and formatting.
- **Black is disabled**: `[tool.black] exclude = ".*"` in repos that have migrated to ruff.

### BIDS Compliance

- All outputs must conform to the [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) specification.
- Use `pybids.BIDSLayout` for querying input datasets.
- Use `DerivativesDataSink` (from the project's interfaces or niworkflows) for writing BIDS-compliant output files.
- Entity names, suffixes, and extensions must match the BIDS specification.

---

## QSIRecon-Specific Instructions

### Project Overview

QSIRecon is a BIDS App for reconstructing and postprocessing diffusion MRI data that has been preprocessed by QSIPrep. It supports:
- Diffusion model fitting (DTI, DKI, MAPMRI, GQI, NODDI via AMICO)
- Tractography and bundle segmentation (MRtrix3, DSI Studio, TORTOISE, pyAFQ)
- Scalar mapping to atlases
- Connectivity matrix generation
- ODF conversion between formats

### Repository Details

| Item | Value |
|------|-------|
| Package name | `qsirecon` |
| Default branch | `main` |
| Entry point | `qsirecon.cli.run:main` |
| Python requirement | `>=3.10` |
| Build backend | hatchling + hatch-vcs + cython + numpy |
| Linter | **flake8 + black + isort** (migration to ruff pending) |
| Pre-commit | **None** (to be added) |
| Tox | **None** (to be added) |
| Docker base | `pennlinc/qsirecon_build:<ver>` |
| Dockerfile | Multi-stage (wheelstage defined but unused in COPY) |

### Key Directories

- `qsirecon/workflows/recon/`: Reconstruction workflow modules organized by tool (amico, dipy, dsi_studio, mrtrix, pyafq, tortoise)
- `qsirecon/interfaces/`: Nipype interfaces for AMICO, DIPY, DSI Studio, MRtrix3, pyAFQ, TORTOISE, and scalar mapping
- `qsirecon/data/`: YAML-based reconstruction workflow specifications (44 `.yaml` files defining reconstruction pipelines)
- `qsirecon/cli/convertODFs.py`: CLI tool for converting between ODF formats (MIF <-> FIB)
- `qsirecon/cli/group_report.py`: Aggregates individual reports into group-level reports

### IMPORTANT: Current Linting State

QSIRecon has **not yet migrated to ruff**. It currently uses:
- `black` for formatting (line-length 99, target Python 3.8, **double quotes**)
- `isort` for import sorting (profile "black")
- `flake8` for linting (configured in `[tool.flake8]` in `pyproject.toml`)

The CI lint workflow (`.github/workflows/lint.yml`) still runs `flake8`, not ruff.

**When writing new code for qsirecon**:
- Use **double quotes** (the current black default) until the ruff migration is complete.
- Follow `isort` "black" profile for import ordering.
- The migration to ruff + single quotes is a planned maintenance task (see roadmap below).

### Reconstruction Workflow Specifications

QSIRecon uses YAML files in `qsirecon/data/` to define reconstruction pipelines. These YAML files specify:
- Which reconstruction nodes to run
- Node parameters
- How nodes connect to each other

When adding new reconstruction workflows, create a new `.yaml` file following existing patterns and register it in the workflow builder.

### External Tool Dependencies

QSIRecon wraps several external neuroimaging tools:
- **pyAFQ** (`== 2.0`): Automated fiber quantification
- **dmri-amico** (`== 2.0.3`): AMICO implementation of NODDI
- **Ingress2QSIRecon** (`== 0.2.3`): Data ingression from other pipelines
- **DSI Studio**, **MRtrix3**, **TORTOISE**: Called via subprocess interfaces

---

## Cross-Project Development Roadmap

This roadmap covers harmonization work across all four PennLINC BIDS Apps (qsiprep, qsirecon, xcp_d, aslprep) to reduce maintenance burden.

### Phase 1: Bring qsirecon to parity

1. **Migrate qsirecon from flake8+black+isort to ruff** -- copy the `[tool.ruff]` config from xcp_d's `pyproject.toml` and remove `[tool.black]`, `[tool.isort]`, `[tool.flake8]` sections.
2. **Add `.pre-commit-config.yaml` to qsirecon** -- identical to the config used by qsiprep, xcp_d, and aslprep.
3. **Add `tox.ini` to qsirecon** -- copy from qsiprep or xcp_d (they are identical).
4. **Add `.github/dependabot.yml` to qsirecon**.
5. **Reformat qsirecon codebase** -- run `ruff format` to switch from double quotes to single quotes.

### Phase 2: Standardize across all four repos

6. **Rename qsiprep default branch** from `master` to `main` and update `.github/workflows/lint.yml`.
7. **Rename aslprep test extras** from `test` to `tests` for consistency with the other three repos.
8. **Converge on version management** -- recommend the simpler `_version.py` direct-import pattern (used by qsiprep/qsirecon). Migrate xcp_d and aslprep away from `__about__.py`.
9. **Pin the same ruff version** in all four repos' dev dependencies and `.pre-commit-config.yaml`.
10. **Harmonize ruff ignore lists** -- adopt xcp_d's minimal set (`S105`, `S311`, `S603`) as the target; fix suppressed rules in qsiprep and aslprep incrementally.

### Phase 3: Shared infrastructure

11. **Extract a reusable GitHub Actions workflow** for lint + codespell + build checks, hosted in a shared repo (e.g., `PennLINC/.github`).
12. **Standardize Dockerfile patterns** -- adopt multi-stage wheel builds (as qsiprep does) across all four repos.
13. **Create a shared `pennlinc-style` package or cookiecutter template** providing `pyproject.toml` lint/test config, `.pre-commit-config.yaml`, `tox.ini`, and CI workflows.
14. **Evaluate `nipreps-versions` calver** -- the `raw-options = { version_scheme = "nipreps-calver" }` line is commented out in all four repos. Decide whether to adopt it.

