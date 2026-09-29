#!/usr/bin/env bash
#
# Local launcher for the N-Pulse visual interface.
#
#   ./run_local.sh              start on http://127.0.0.1:5001
#   PORT=5002 ./run_local.sh    start on a different port
#   SKIP_INSTALL=1 ./run_local.sh   skip the dependency check (faster restarts)
#
# The script creates a virtual environment, installs the dependencies and
# starts the Flask-SocketIO server.  Do not use `flask run` instead — the
# Socket.IO server has to be started through app.py or the browser will
# connect but never receive any data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 was not found on this machine."
    echo "Install Python 3.12 (recommended) and run this script again."
    exit 1
fi

# The venv lives at the repository root when this script is used inside the
# repo (visual_interface/run_local.sh), and beside the script when the folder
# is used on its own.  Both layouts are supported, and an existing venv always
# wins so nobody ends up with two of them.
if [ -d "$REPO_ROOT/.venv" ]; then
    VENV_DIR="$REPO_ROOT/.venv"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    VENV_DIR="$SCRIPT_DIR/.venv"
elif [ -d "$REPO_ROOT/visual_interface" ]; then
    VENV_DIR="$REPO_ROOT/.venv"
else
    VENV_DIR="$SCRIPT_DIR/.venv"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# Dependencies are checked on every start, not only when the venv is created.
# Otherwise an environment made before requirements.txt changed keeps being
# reused and the app fails later with a confusing ImportError.
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    echo "Checking dependencies ..."
    "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

cd "$SCRIPT_DIR"
export PORT="${PORT:-5001}"

echo "Starting the visual interface on http://127.0.0.1:${PORT}"
echo "Press Ctrl+C to stop."
exec "$VENV_DIR/bin/python" app.py
