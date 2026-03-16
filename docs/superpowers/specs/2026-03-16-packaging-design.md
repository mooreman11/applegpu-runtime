# Packaging Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** CI/CD pipeline, Python wheel builds, Swift binary distribution, install script, TestPyPI validation, version bump.

## Overview

Ship applegpu_runtime as installable artifacts: Python wheels for macOS ARM and Linux aarch64 (Python 3.10-3.13), Swift binaries for gpu-container and gpu-service via GitHub Releases, and an install script. Validate the wheel pipeline via TestPyPI. Enable CI on every push/PR.

## Artifacts

| Artifact | Platform | Format | Distribution |
|----------|----------|--------|--------------|
| applegpu_runtime wheel | macOS ARM64 | `.whl` (cp310-cp313) | GitHub Release, TestPyPI |
| applegpu_runtime wheel | Linux aarch64 | `.whl` (cp310-cp313) | GitHub Release, TestPyPI |
| gpu-container | macOS ARM64 | binary | GitHub Release |
| gpu-service | macOS ARM64 | binary | GitHub Release |
| install.sh | any | shell script | Raw GitHub URL |

Total: 8 wheels (4 Python versions × 2 platforms) + 2 binaries + 1 script.

## CI Workflow (`.github/workflows/ci.yml`)

Triggers: push to `main`, pull requests to `main`.

**Jobs:**

1. **test-rust** (`macos-14`): `cargo test -p applegpu-core` (not `--workspace` — avoids PyO3 cdylib link failure)
2. **test-swift** (`macos-14`): `cd swift && swift test`
3. **test-python** (`macos-14`): `uv sync && uv run maturin develop && uv run pytest -v`

Note: `macos-14` runners are Apple Silicon (M-series). Metal GPU tests run for real — all 310 Rust tests and 334 Python tests pass locally. If CI runners lack GPU access, we'll gate GPU-dependent tests behind a feature flag or `#[ignore]` attribute.

## Release Workflow (`.github/workflows/release.yml`)

Triggers: push tag matching `v*` (e.g., `v0.9.0`).

**Jobs:**

1. **build-wheels-macos** (`macos-14`):
   - Install Rust, uv, maturin
   - `maturin build --release -i python3.10 python3.11 python3.12 python3.13`
   - Upload wheels as artifacts

2. **build-wheels-linux** (`macos-14`):
   - Install Rust, uv, maturin, zig (for cross-compilation)
   - `maturin build --release --target aarch64-unknown-linux-gnu --zig -i python3.10 python3.11 python3.12 python3.13`
   - Upload wheels as artifacts

3. **build-binaries** (`macos-14`):
   - `cargo build -p applegpu-service --release`
   - `cd swift/GPUContainer && swift build -c release`
   - Upload `gpu-service` and `gpu-container` binaries as artifacts

4. **release** (needs: build-wheels-macos, build-wheels-linux, build-binaries):
   - Download all artifacts
   - Create GitHub Release with tag name, auto-generated changelog
   - Attach all wheels and binaries
   - `twine upload --repository testpypi` for all wheels (using `TESTPYPI_TOKEN` secret)

## Install Script (`install.sh`)

Located at repo root. Usage:
```bash
curl -fsSL https://raw.githubusercontent.com/mooreman11/applegpu-runtime/main/install.sh | sh
```

Behavior:
1. Detect latest release via GitHub API (`/repos/mooreman11/applegpu-runtime/releases/latest`)
2. Detect platform (must be macOS ARM64, error otherwise)
3. Download `gpu-container` and `gpu-service` binaries from release assets
4. Install to `~/.local/bin/` by default, or `/usr/local/bin/` if `--system` flag
5. Add `~/.local/bin` to PATH in `.zshrc` / `.bashrc` if not present
6. Verify installation: `gpu-service --version && gpu-container --version`

## Version Strategy

Bump from `0.1.0` to `0.9.0` for this release:
- `Cargo.toml` workspace version: `0.9.0`
- `pyproject.toml` version: managed by maturin (reads from Cargo.toml)
- Git tag: `v0.9.0`

Add `--version` flag to both `gpu-service` and `gpu-container` binaries.

## Repository Secrets Required

| Secret | Purpose |
|--------|---------|
| `TESTPYPI_TOKEN` | API token for TestPyPI upload |

No PyPI token needed yet — TestPyPI only for validation.

## Not Included

- **PyPI publishing** — TestPyPI only. Real PyPI after validation.
- **Container base image** — deferred. Docker bind-mount workflow works.
- **Homebrew tap** — can add later with a formula that downloads from GitHub Releases.
- **Windows/x86 builds** — Apple Silicon only project.
- **Signing/notarization** — macOS binaries are unsigned. Can add Apple Developer ID signing later.

## Success Criteria

1. `git tag v0.9.0 && git push --tags` triggers release workflow
2. GitHub Release appears with 8 wheels + 2 binaries attached
3. `pip install --index-url https://test.pypi.org/simple/ applegpu_runtime` works
4. `curl ... | sh` installs working gpu-container and gpu-service
5. CI runs on every PR and push to main
