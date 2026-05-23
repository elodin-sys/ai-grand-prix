# AI Grand Prix Playground: helper tasks only.
#
# The main workflow is intentionally direct and Elodin-native:
#
#   elodin editor sim/main.py
#
# This file only keeps setup/test conveniences that are not core Elodin commands.

default:
    @just --list

# Install Elodin CLI binary and sync Python dependencies.
setup:
    bash scripts/install_elodin.sh
    uv sync

# Initialize/update the Betaflight submodule and build SITL.
betaflight:
    bash scripts/fetch_betaflight.sh
    bash scripts/build_betaflight.sh

# Run Python unit tests.
test:
    uv run pytest

# Run the takeoff smoke verifier against the latest sim output.
verify:
    uv run python scripts/verify_takeoff.py

# Remove generated simulation databases and exports.
clean:
    rm -rf betaflight_db[0-9]* dbs/
