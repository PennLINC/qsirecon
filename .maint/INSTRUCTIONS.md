# Maintenance instructions for QSIRecon

Run all commands from the repository root unless noted otherwise.

---

## Updating runtime dependencies in Dockerfile.base

`Dockerfile.base` contains non-Python runtime dependencies (for example,
neuroimaging binaries and OS-level packages). These are managed separately from
pixi-managed dependencies.

1. **Edit `Dockerfile.base`** to update versions, URLs, or package lists.
2. **Bump the base image date tag** in `Dockerfile`:

   ```dockerfile
   ARG BASE_IMAGE=pennlinc/qsirecon-base:<YYYYMMDD>
   ```

3. **Commit and push.** CircleCI `image_prep` checks whether the configured
   `BASE_IMAGE` tag exists in Docker Hub.
   - If missing, it builds `Dockerfile.base` and pushes both the date tag and
     `latest`.
   - If present, base-image build is skipped.
4. **Verify** CI succeeds and the new base image appears at
   `pennlinc/qsirecon-base:<YYYYMMDD>`.

For local testing:

```bash
docker build -f Dockerfile.base -t pennlinc/qsirecon-base:$(date +%Y%m%d) .
docker build --target qsirecon -t pennlinc/qsirecon:dev .
```

---

## Updating individual Python dependencies in pyproject.toml

QSIRecon uses two dependency sections in `pyproject.toml`:

| Section | Managed by | Examples |
|---------|-----------|----------|
| `[project.dependencies]` | PyPI (pip) | nibabel, nipype, niworkflows |
| `[tool.pixi.dependencies]` | conda (pixi) | python, numpy, ANTs/FSL toolchain |

Both resolve into `pixi.lock`.

1. **Edit version specifiers** in `pyproject.toml`.
2. **Commit and open a PR.**
3. **`pixi-lock.yml` runs on every `pull_request`.** It checks whether the PR
   (`base...HEAD`, not only the latest commit) touched `pyproject.toml` or
   `pixi.lock`, and only then runs the lockfile steps.
4. **Lockfile behavior depends on PR source:**
   - **Same-repo PR branches:** the workflow runs `pixi lock --check`, then restores
     the committed `pixi.lock` (the check rewrites the workspace package's own
     version, which changes with every commit, even when it passes). Only if the
     check failed does it regenerate `pixi.lock` with `datalad run ... -- pixi lock`
     and push that commit to the PR branch. GitHub holds the checks for the bot's
     commit as "action required" until a maintainer approves them.
   - **Fork PR branches:** the workflow only checks the lockfile (`pixi lock --check`),
     with read-only permissions, and does **not** push. If the check fails, update
     `pixi.lock` manually on Linux and push from the contributor branch.
5. **Review lockfile changes locally** if needed (diff rendering may be limited in UI).

> **Note:** The lockfile targets Linux; regenerate `pixi.lock` in a Linux
> environment.

---

## Updating all Python dependencies with pixi update

`pixi update` resolves all dependencies to the newest versions allowed by
`pyproject.toml` constraints and rewrites `pixi.lock`. Run this on Linux.

1. Clone and create a branch.
2. Install pixi (if needed).
3. Run:

   ```bash
   pixi update
   ```

   To update one package only:

   ```bash
   pixi update <package-name>
   ```

4. Review, commit, and push the updated `pixi.lock`.
5. Open a PR and let CI validate compatibility.

---

## CI workflow trigger conditions

This section describes what causes lockfile updates, image rebuilds, and deploys.

### GitHub Action: `.github/workflows/pixi-lock.yml`

The workflow triggers on every `pull_request` event and runs one of two jobs,
depending on where the PR branch lives, plus an aggregate job:

- **`validate-lockfile`** (fork PRs only; read-only, no persisted credentials)
  - Always runs `Checkout pull request` and `Check pull request for dependency edits`.
  - If the PR touched `pyproject.toml` or `pixi.lock`, runs `prefix-dev/setup-pixi`
    and `pixi lock --check`. It never pushes.
- **`update-lockfile`** (same-repo PRs only; the only job with `contents: write`)
  - Always runs `Checkout pull request` (with `persist-credentials: false`) and
    `Check pull request for dependency edits`.
  - If the PR touched `pyproject.toml` or `pixi.lock`, runs `prefix-dev/setup-pixi`
    and `Check lockfile`, which runs `pixi lock --check`, records `stale=true|false`,
    and resets `pixi.lock` with `git checkout -- pixi.lock`.
  - If the lockfile is stale, runs `Set git identity`, `Install the latest version
    of uv`, `Update lockfile` (`uvx datalad run -i pixi.lock -i pyproject.toml
    -o pixi.lock -- pixi lock`), and `Push updated lockfile` (non-forced). The push
    step is the only one that receives the token, passed as an `http.extraheader`
    on the `git push` command line.
- **`check-lockfile`** (always)
  - Fails unless exactly the applicable job above succeeded and the other was skipped.
    This is the job to require as a status check.

Practical implication: editing non-dependency files still triggers the workflow,
but the pixi steps are skipped.

Do not move `pixi lock --check` back under `datalad run`. The check exits 0 but
still rewrites the workspace package's hatch-vcs version in `pixi.lock`, so
running it under DataLad committed and pushed that rewrite on every run, and each
bot push queued another run (#396).

### CircleCI: `.circleci/config.yml` (hybrid model)

QSIRecon uses a hybrid model:

- `image_prep` builds/prepares **test infrastructure** only:
  - checks/builds base image if needed
  - builds/reuses `pennlinc/qsirecon:test`
  - pushes test image to local registry cache
- `deploy_docker` builds/pushes **production image** only on deploy paths:
  - runs on `main` and tags
  - requires `deployable`
  - `deployable` requires full integration matrix + `image_prep`

This keeps strict deploy gating while avoiding production builds on routine PR CI.

#### Branch/tag filters

- `get_data` and `merge_coverage` ignore branches matching `docs?/.*` and `tests?/.*`.
- `deployable` and `deploy_docker` run only on `main` and tags.
- `image_prep` has no branch-ignore filter; it runs wherever the workflow runs.

#### Build cache and marker behavior

Primary cache key:

```text
build-v4-{{ checksum "Dockerfile" }}-{{ checksum "pixi.lock" }}-{{ .Revision }}
```

The key includes the commit revision, so test-image cache reuse is scoped to the
same commit and does not cross into later commits.

When `/tmp/images/imageprep-success.marker` is restored:

- `BUILD_TEST_IMAGE=0` and test-image rebuild is skipped.

When marker is missing:

- `BUILD_TEST_IMAGE=1` and test image is rebuilt.

Base-image behavior is independent from marker: CircleCI checks whether
`BASE_IMAGE` in `Dockerfile` exists in Docker Hub; if missing, `Dockerfile.base`
is rebuilt and pushed.

Production image behavior:

- Built in `deploy_docker` (not `image_prep`)
- Verified with `qsirecon --version`
- Pushed as `unstable` on `main`; additionally pushed as `latest` and
  `<CIRCLE_TAG>` on tagged builds

Test-image guard behavior:

- `image_prep` builds the test image with `VCS_REF=$(git rev-parse --short HEAD)`.
- The test image is labeled with `org.opencontainers.image.revision`.
- Before tests run, CircleCI compares that label to the checkout SHA and fails
  early on mismatch to prevent stale-image test runs.

#### File edits vs CI image behavior

- **Edit `Dockerfile` and change `ARG BASE_IMAGE=...` to a new/missing tag**
  - Base image: **rebuilt** (manifest missing)
  - Test image: **rebuilt** (cache key changes)
  - Production image: built only in `deploy_docker` on `main`/tags
- **Edit `Dockerfile` without changing base tag**
  - Base image: rebuilt only if configured tag is missing
  - Test image: **rebuilt** (cache key changes)
  - Production image: deploy-path only
- **Edit `Dockerfile.base` only**
  - Base image: **not rebuilt automatically** if configured base tag already exists
  - Test image: typically reused (cache key unchanged)
  - To force base rebuild, bump `ARG BASE_IMAGE=...` in `Dockerfile`
- **Edit `pixi.lock` only**
  - Base image: unchanged unless configured tag is missing
  - Test image: **rebuilt** (cache key changes)
  - Production image: deploy-path only
- **Edit neither `Dockerfile` nor `pixi.lock`**
  - Test image usually reused when marker/cache restore succeeds
  - Base image still checked for remote existence
  - Production image unchanged unless deploy job runs

---

## Releasing a new version

### 1. Prepare release branch

- Ensure release branch is up to date and CI is green.
- Confirm target tag does not already exist:

  ```bash
  git tag -l
  ```

### 2. Update `CITATION.cff`

- Set `version` to the new release version.
- Set `date-released` to `YYYY-MM-DD`.

### 3. (Optional) Refresh authorship metadata

If maintaining `.zenodo.json`:

```bash
python .maint/update_authors.py zenodo
```

### 4. Base image update (if needed)

- Runtime image is built `FROM pennlinc/qsirecon-base:<YYYYMMDD>`.
- Base image is built from `Dockerfile.base`.
- To publish a new base image, bump the date tag in `Dockerfile` and push.

### 5. Commit and push release-prep changes

- Commit release-related edits (for example `CITATION.cff`, base tag changes).
- Push the release branch.

### 6. Create and push annotated tag

```bash
git tag -a <version> -m "Release <version>"
git push origin <version>
```

### 7. CI deploy flow on release tags

On tagged builds, CircleCI runs:

1. `image_prep` (test-image prep and base check)
2. integration test matrix
3. `deployable` gate
4. `deploy_docker` (production build + push)

Docker tags pushed when authenticated:

- `unstable` (always on deploy job)
- `latest` and `<version>` (when `CIRCLE_TAG` is set)

### 8. Create GitHub Release

- Draft release from pushed tag.
- Use auto-generated notes or curate from changelog.
- Publish release.

### Release checklist

- [ ] `CITATION.cff` version/date updated
- [ ] (Optional) authorship/Zenodo metadata refreshed
- [ ] (If needed) base image date tag bumped in `Dockerfile`
- [ ] Changes committed and pushed
- [ ] Annotated tag created and pushed
- [ ] GitHub Release drafted and published
