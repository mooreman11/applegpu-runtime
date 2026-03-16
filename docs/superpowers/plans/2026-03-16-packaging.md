# Packaging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship applegpu_runtime as installable Python wheels (macOS ARM + Linux aarch64, Python 3.10-3.13) with Swift binary distribution via GitHub Releases, CI on every push/PR, and TestPyPI validation.

**Architecture:** Two GitHub Actions workflows: CI (test on every push/PR using M2 Pro GPU runners) and Release (build wheels + binaries on tag push, create GitHub Release, upload to TestPyPI). Install script downloads binaries from latest release. Version bumped to 0.8.0 with dynamic pyproject.toml versioning.

**Tech Stack:** GitHub Actions, maturin, PyO3/maturin-action, zig (cross-compilation), twine (TestPyPI), Swift ArgumentParser

**Spec:** `docs/superpowers/specs/2026-03-16-packaging-design.md`

---

## Chunk 1: Version Bump + CI

### Task 1: Version bump to 0.8.0

**Files:**
- Modify: `Cargo.toml` (workspace version)
- Modify: `pyproject.toml` (dynamic version)

- [ ] **Step 1: Bump workspace version**

In `Cargo.toml`, change:
```toml
[workspace.package]
version = "0.8.0"
```

- [ ] **Step 2: Make pyproject.toml version dynamic**

In `pyproject.toml`, replace `version = "0.1.0"` with dynamic versioning:
```toml
[project]
name = "applegpu_runtime"
dynamic = ["version"]
requires-python = ">=3.10"
```

Remove the `version = "0.1.0"` line entirely. Maturin reads the version from `Cargo.toml` when `version` is in `dynamic`.

- [ ] **Step 3: Verify maturin resolves version**

Run: `uv run maturin develop 2>&1 | grep "Built wheel"`
Expected: Output contains `applegpu_runtime-0.8.0`

- [ ] **Step 4: Verify Python reports correct version**

Run: `uv run python -c "import applegpu_runtime; print(applegpu_runtime.version())"`
Expected: `0.8.0`

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml pyproject.toml
git commit -m "chore: bump version to 0.8.0, dynamic pyproject.toml version"
```

### Task 2: Add --version flag to binaries

**Files:**
- Modify: `crates/gpu-service/src/main.rs`
- Modify: `swift/GPUContainer/Sources/GPUContainer.swift`

- [ ] **Step 1: Add --version to gpu-service**

At the top of `main()` in `crates/gpu-service/src/main.rs`, before any other logic:

```rust
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--version" {
        println!("gpu-service {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    // ... existing code ...
```

- [ ] **Step 2: Test gpu-service --version**

Run: `cargo run -p applegpu-service -- --version`
Expected: `gpu-service 0.8.0`

- [ ] **Step 3: Add --version to gpu-container**

In `swift/GPUContainer/Sources/GPUContainer.swift`, add version to the command configuration:

```swift
@main
struct GPUContainer: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gpu-container",
        abstract: "Run Linux containers with Metal GPU access on Apple Silicon",
        version: "0.8.0",
        subcommands: [Run.self, Stop.self, List.self, Status.self, Service.self]
    )
}
```

ArgumentParser automatically handles `--version` when `version` is set.

- [ ] **Step 4: Test gpu-container --version**

Run: `cd swift/GPUContainer && swift run gpu-container --version`
Expected: `0.8.0`

- [ ] **Step 5: Commit**

```bash
git add crates/gpu-service/src/main.rs swift/GPUContainer/Sources/GPUContainer.swift
git commit -m "feat: add --version flag to gpu-service and gpu-container"
```

### Task 3: Fix and enable CI workflow

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Rewrite ci.yml**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  test-rust:
    name: Rust Tests
    runs-on: macos-15-xlarge
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test -p applegpu-core -p applegpu-wire -p applegpu-client -p applegpu-service

  test-swift:
    name: Swift Tests
    runs-on: macos-15-xlarge
    steps:
      - uses: actions/checkout@v4
      - run: cd swift && swift test

  test-python:
    name: Python Tests
    runs-on: macos-15-xlarge
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run maturin develop
      - run: uv run pytest -v
```

Key changes from existing:
- `runs-on: macos-15-xlarge` (M2 Pro with Metal GPU)
- `cargo test -p ...` instead of `--workspace`
- Added `uv run maturin develop` before pytest
- Removed `|| echo` fallback from Swift tests
- Enabled push to `main` and pull_request triggers

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: enable CI with M2 Pro GPU runners, fix test commands"
```

---

## Chunk 2: Release Workflow

### Task 4: Create release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write release.yml**

```yaml
name: Release

on:
  push:
    tags: ['v*']

permissions:
  contents: write

env:
  CARGO_TERM_COLOR: always

concurrency:
  group: release
  cancel-in-progress: false

jobs:
  build-wheels-macos:
    name: Build macOS Wheels
    runs-on: macos-15-xlarge
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: |
            3.10
            3.11
            3.12
            3.13
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release -i python3.10 python3.11 python3.12 python3.13
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-macos
          path: target/wheels/*.whl

  build-wheels-linux:
    name: Build Linux aarch64 Wheels
    runs-on: macos-15-xlarge
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: |
            3.10
            3.11
            3.12
            3.13
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: goto-bus-stop/setup-zig@v2
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --target aarch64-unknown-linux-gnu --zig -i python3.10 python3.11 python3.12 python3.13
          manylinux: auto
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-linux
          path: target/wheels/*.whl

  build-binaries:
    name: Build Swift + Rust Binaries
    runs-on: macos-15-xlarge
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build gpu-service
        run: cargo build -p applegpu-service --release
      - name: Build gpu-container
        run: cd swift/GPUContainer && swift build -c release
      - name: Collect binaries
        run: |
          mkdir -p dist
          cp target/release/gpu-service dist/
          cp swift/GPUContainer/.build/release/gpu-container dist/
          cd dist && shasum -a 256 * > checksums.txt
      - uses: actions/upload-artifact@v4
        with:
          name: binaries
          path: dist/*

  release:
    name: Create Release
    needs: [build-wheels-macos, build-wheels-linux, build-binaries]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          path: artifacts
      - name: Collect all artifacts
        run: |
          mkdir -p dist
          cp artifacts/wheels-macos/*.whl dist/
          cp artifacts/wheels-linux/*.whl dist/
          cp artifacts/binaries/* dist/
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: dist/*
          generate_release_notes: true
      - name: Publish to TestPyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TESTPYPI_TOKEN }}
        run: |
          pip install twine
          twine upload --repository testpypi dist/*.whl
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add release workflow — wheels, binaries, GitHub Release, TestPyPI"
```

### Task 5: Relax GPUContainer macOS platform version

**Files:**
- Modify: `swift/GPUContainer/Package.swift`

The GPUContainer currently requires macOS 26.0, but CI runners run macOS 15. The ContainerRunner.swift uses `#available(macOS 26, *)` guards, so the binary will compile on older macOS — it just won't use the Containerization framework at runtime.

- [ ] **Step 1: Change platform version**

In `swift/GPUContainer/Package.swift`, change:
```swift
platforms: [.macOS(.v14)],
```

This allows building on macOS 14+ runners while keeping runtime guards for macOS 26+ features.

- [ ] **Step 2: Verify build**

Run: `cd swift/GPUContainer && swift build -c release`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add swift/GPUContainer/Package.swift
git commit -m "fix: relax GPUContainer platform to macOS 14 for CI compatibility"
```

---

## Chunk 3: Install Script + Validation

### Task 6: Create install script

**Files:**
- Create: `install.sh`

- [ ] **Step 1: Write install.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="mooreman11/applegpu-runtime"
INSTALL_DIR="${HOME}/.local/bin"

# Parse args
for arg in "$@"; do
    case "$arg" in
        --system) INSTALL_DIR="/usr/local/bin" ;;
        --help|-h)
            echo "Usage: install.sh [--system]"
            echo "  --system  Install to /usr/local/bin (requires sudo)"
            echo "  Default:  Install to ~/.local/bin"
            exit 0
            ;;
    esac
done

# Check platform
ARCH=$(uname -m)
OS=$(uname -s)
if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then
    echo "Error: applegpu-runtime requires macOS on Apple Silicon (arm64)"
    echo "Detected: $OS $ARCH"
    exit 1
fi

# Get latest release
echo "Fetching latest release..."
RELEASE_URL=$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"browser_download_url"' | head -1 | cut -d'"' -f4)
if [ -z "$RELEASE_URL" ]; then
    echo "Error: Could not find latest release"
    exit 1
fi
TAG=$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)
echo "Latest release: $TAG"

# Download binaries
RELEASE_BASE="https://github.com/${REPO}/releases/download/${TAG}"
mkdir -p "$INSTALL_DIR"

echo "Downloading gpu-service..."
curl -sL "${RELEASE_BASE}/gpu-service" -o "${INSTALL_DIR}/gpu-service"
chmod +x "${INSTALL_DIR}/gpu-service"

echo "Downloading gpu-container..."
curl -sL "${RELEASE_BASE}/gpu-container" -o "${INSTALL_DIR}/gpu-container"
chmod +x "${INSTALL_DIR}/gpu-container"

# Download and verify checksums
echo "Verifying checksums..."
CHECKSUMS=$(curl -sL "${RELEASE_BASE}/checksums.txt")
cd "$INSTALL_DIR"
echo "$CHECKSUMS" | grep -E "gpu-service|gpu-container" | shasum -a 256 -c - || {
    echo "Error: Checksum verification failed!"
    exit 1
}

# Add to PATH if needed
SHELL_RC=""
if [ -f "${HOME}/.zshrc" ]; then
    SHELL_RC="${HOME}/.zshrc"
elif [ -f "${HOME}/.bashrc" ]; then
    SHELL_RC="${HOME}/.bashrc"
fi

if [ -n "$SHELL_RC" ] && ! grep -q "${INSTALL_DIR}" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# applegpu-runtime" >> "$SHELL_RC"
    echo "export PATH=\"${INSTALL_DIR}:\$PATH\"" >> "$SHELL_RC"
    echo "Added ${INSTALL_DIR} to PATH in ${SHELL_RC}"
fi

# Verify
echo ""
echo "Installed:"
"${INSTALL_DIR}/gpu-service" --version
"${INSTALL_DIR}/gpu-container" --version
echo ""
echo "Done! Restart your shell or run: export PATH=\"${INSTALL_DIR}:\$PATH\""
```

- [ ] **Step 2: Make executable**

Run: `chmod +x install.sh`

- [ ] **Step 3: Commit**

```bash
git add install.sh
git commit -m "feat: add install.sh for gpu-container + gpu-service binary installation"
```

### Task 7: Push and validate

- [ ] **Step 1: Push all changes**

```bash
git push origin main
```

Verify CI runs on push and passes.

- [ ] **Step 2: Create TestPyPI account and token**

1. Go to https://test.pypi.org/account/register/
2. Create account (if not already)
3. Go to Account Settings → API Tokens → Add API Token
4. Scope: project `applegpu_runtime` (or entire account for first upload)
5. Add as GitHub secret: repo Settings → Secrets → `TESTPYPI_TOKEN`

- [ ] **Step 3: Tag and release**

```bash
git tag v0.8.0
git push origin v0.8.0
```

This triggers the release workflow. Monitor at `https://github.com/mooreman11/applegpu-runtime/actions`.

- [ ] **Step 4: Verify GitHub Release**

Check `https://github.com/mooreman11/applegpu-runtime/releases/tag/v0.8.0`:
- 8 wheel files attached (4 macOS + 4 Linux)
- 2 binary files (gpu-service, gpu-container)
- checksums.txt

- [ ] **Step 5: Verify TestPyPI**

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ applegpu_runtime
python -c "import applegpu_runtime; print(applegpu_runtime.version())"
```

Expected: `0.8.0`

- [ ] **Step 6: Test install script (after release exists)**

```bash
curl -fsSL https://raw.githubusercontent.com/mooreman11/applegpu-runtime/v0.8.0/install.sh | sh
gpu-service --version
gpu-container --version
```

Expected: Both print `0.8.0`

- [ ] **Step 7: Commit milestone**

```bash
git commit --allow-empty -m "milestone: v0.8.0 released — wheels, binaries, TestPyPI, install script"
```
