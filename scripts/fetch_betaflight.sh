#!/usr/bin/env bash
# Initialize/update the Betaflight git submodule.
#
# This mirrors the upstream Elodin example workflow: the repo owns a Betaflight
# submodule at ./betaflight, and contributors run this once after cloning.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Initializing Betaflight submodule..."
git submodule update --init --recursive --depth 1 betaflight

echo "Done. Now run: bash scripts/build_betaflight.sh"
