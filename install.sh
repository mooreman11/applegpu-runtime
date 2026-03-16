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
RELEASE_JSON=$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest")
TAG=$(echo "$RELEASE_JSON" | grep '"tag_name"' | cut -d'"' -f4)
if [ -z "$TAG" ]; then
    echo "Error: Could not find latest release"
    exit 1
fi
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
