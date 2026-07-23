#!/usr/bin/env bash
# One-time environment setup for passive-liveness.
# Run once from the passive-liveness/ directory:
#   bash setup.sh

set -e

PYTHON=${PYTHON:-python3}
VENV_DIR="venv"

echo "==> Python: $($PYTHON --version)"

if [ -d "$VENV_DIR" ]; then
    echo "==> venv already exists — skipping creation"
else
    echo "==> Creating virtual environment in ./${VENV_DIR}/"
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate the venv so that `pip`/`python` below refer to the venv's own tools.
# (Installs land inside ./venv, never in the global Python.)
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Safety check: make sure the venv actually has pip (missing python3-venv on
# some distros creates a venv without it).
if ! python -m pip --version >/dev/null 2>&1; then
    echo "ERROR: pip not found in the venv. On Debian/Ubuntu install it with:" >&2
    echo "       sudo apt install python3-venv python3-pip" >&2
    exit 1
fi

echo "==> Installing dependencies into $(python -c 'import sys; print(sys.prefix)')"
python -m pip install --upgrade --quiet pip
python -m pip install --quiet -r requirements.txt

echo ""
echo "Done. Versions installed:"
python -m pip show torch opencv-python numpy | grep -E "^(Name|Version):"

echo ""
echo "The venv is active in THIS shell only if you ran 'source setup.sh'."
echo "For a normal 'bash setup.sh' run, activate it yourself with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Quick test:"
echo "  python predict.py --image images/sample/image_T1.jpg --save"
