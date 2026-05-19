#!/usr/bin/env bash
# Install the Elodin CLI binary (editor + run) for the current platform.
#
# Wraps the official installer for the v0.17.2 release.

set -euo pipefail

VERSION="${ELODIN_VERSION:-v0.17.2}"
INSTALLER_URL="https://github.com/elodin-sys/elodin/releases/download/${VERSION}/elodin-installer.sh"
DB_INSTALLER_URL="https://github.com/elodin-sys/elodin/releases/download/${VERSION}/elodin-db-installer.sh"

echo "Installing Elodin CLI ${VERSION}..."
curl --proto '=https' --tlsv1.2 -LsSf "${INSTALLER_URL}" | sh

echo "Installing Elodin DB CLI ${VERSION}..."
curl --proto '=https' --tlsv1.2 -LsSf "${DB_INSTALLER_URL}" | sh

echo
echo "Done. Verify with:"
echo "  elodin --version"
echo "  elodin-db --version"
