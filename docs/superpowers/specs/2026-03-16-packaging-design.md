# Packaging Design

**Date:** 2026-03-16
**Updated:** 2026-03-16 (post review)
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
| install.sh | any | shell script | Raw GitHub URL (pinned to tag) |

Total: 8 wheels (4 Python versions × 2 platforms) + 2 binaries + 1 script.

## CI Workflow (`.github/workflows/ci.yml`)

Triggers: push to `main`, pull requests to `main`.

Uses `macos-15-xlarge` runners — M2 Pro with 8-core GPU and Metal hardware acceleration enabled by default ($0.16/min). All GPU tests run for real, no gating needed.

**Jobs:**

1. **test-rust** (`macos-15-xlarge`):
   - `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2`
   - `cargo test -p applegpu-core -p applegpu-wire -p applegpu-client -p applegpu-service` (not `--workspace` — avoids PyO3 cdylib link failure)

2. **test-swift** (`macos-15-xlarge`):
   - `cd swift && swift test` (fail the job on failure — remove existing `|| echo` fallback)

3. **test-python** (`macos-15-xlarge`):
   - `astral-sh/setup-uv@v5`
   - `uv sync && uv run maturin develop && uv run pytest -v`

## Release Workflow (`.github/workflows/release.yml`)

Triggers: push tag matching `v*` (e.g., `v0.8.0`).

Uses `PyO3/maturin-action` for wheel builds (handles interpreter discovery, cross-compilation, caching).

**Jobs:**

1. **build-wheels-macos** (`macos-15-xlarge`):
   - `actions/setup-python@v5` with `python-version:` as YAML block scalar (`3.10`, `3.11`, `3.12`, `3.13` each on its own line)
   - `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2`
   - `PyO3/maturin-action@v1` with `command: build`, `args: --release -i python3.10 python3.11 python3.12 python3.13`
   - `actions/upload-artifact@v4` to upload `target/wheels/*.whl`

2. **build-wheels-linux** (`macos-15-xlarge`):
   - `actions/setup-python@v5` with all 4 Python versions (same as macOS job)
   - `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2`
   - `goto-bus-stop/setup-zig@v2`
   - `PyO3/maturin-action@v1` with `command: build`, `args: --release --target aarch64-unknown-linux-gnu --zig -i python3.10 python3.11 python3.12 python3.13`, `manylinux: auto`
   - Maturin uses the host Python interpreters to determine correct wheel tags for cross-compilation.

3. **build-binaries** (`macos-15-xlarge`):
   - `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2`
   - `cargo build -p applegpu-service --release`
   - `cd swift/GPUContainer && swift build -c release`
   - Collect binaries from:
     - `target/release/gpu-service`
     - `swift/GPUContainer/.build/release/gpu-container`
   - `actions/upload-artifact@v4`

4. **release** (needs: build-wheels-macos, build-wheels-linux, build-binaries):
   - `actions/download-artifact@v4` to collect all artifacts
   - `softprops/action-gh-release@v2` to create GitHub Release with auto-generated changelog, attach all wheels + binaries + SHA256 checksums
   - `pip install twine && twine upload --repository testpypi dist/*.whl` (using `TESTPYPI_TOKEN` secret)

## Install Script (`install.sh`)

Located at repo root. Usage:
```bash
curl -fsSL https://raw.githubusercontent.com/mooreman11/applegpu-runtime/v0.8.0/install.sh | sh
```

Note: URL pinned to release tag, not `main`.

Behavior:
1. `set -euo pipefail` at top
2. Detect latest release via GitHub API (`/repos/mooreman11/applegpu-runtime/releases/latest`)
3. Detect platform (must be macOS ARM64, error otherwise)
4. Download `gpu-container` and `gpu-service` binaries from release assets
5. Verify SHA256 checksums against values in release notes
6. Install to `~/.local/bin/` by default, or `/usr/local/bin/` if `--system` flag
7. Add `~/.local/bin` to PATH in `.zshrc` / `.bashrc` if not present
8. Verify installation: `gpu-service --version && gpu-container --version`

## Version Strategy

Bump from `0.1.0` to `0.8.0` for this release (matches roadmap — "Tag v0.8.0"):
- `Cargo.toml` workspace version: `0.8.0`
- `pyproject.toml`: remove hardcoded `version = "0.1.0"`, use dynamic versioning:
  ```toml
  [project]
  dynamic = ["version"]
  ```
  Maturin reads version from `Cargo.toml` automatically when `version` is dynamic.
- Git tag: `v0.8.0`

Add `--version` flag to both binaries:
- `gpu-service`: add `std::env::args()` check for `--version`, print version from `env!("CARGO_PKG_VERSION")`
- `gpu-container`: add `--version` subcommand to the Swift ArgumentParser

## Repository Secrets Required

| Secret | Purpose |
|--------|---------|
| `TESTPYPI_TOKEN` | Project-scoped API token for TestPyPI upload |

No PyPI token needed yet — TestPyPI only for validation.

## Not Included

- **PyPI publishing** — TestPyPI only. Real PyPI after validation.
- **Container base image** — deferred. Docker bind-mount workflow works.
- **Homebrew tap** — can add later with a formula that downloads from GitHub Releases.
- **Windows/x86 builds** — Apple Silicon only project.
- **Signing/notarization** — macOS binaries are unsigned. Can add Apple Developer ID signing later.
- **GPUContainer macOS 26 platform requirement** — `swift/GPUContainer/Package.swift` specifies `.macOS("26.0")`. The `macos-15-xlarge` runner runs macOS 15. Must relax platform version to `.macOS(.v14)` before the Swift binary can build in CI, or wait for macOS 26 runners.
- **Source distribution (sdist)** — wheels only for now. Can add sdist later.

## Success Criteria

1. `git tag v0.8.0 && git push --tags` triggers release workflow
2. GitHub Release appears with 8 wheels + 2 binaries + checksums attached
3. `pip install --index-url https://test.pypi.org/simple/ applegpu_runtime` works
4. `curl ... | sh` installs working gpu-container and gpu-service
5. CI runs on every PR and push to main on M2 Pro GPU runners — all tests pass
