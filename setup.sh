#!/usr/bin/env bash
# One-time environment setup for passive-liveness.
# Run once from the passive-liveness/ directory:
#   bash setup.sh
#
# Note: this script does NOT activate the venv (and is safe to `source`).
# Dependencies are installed straight into ./venv via the venv's own pip.

# Only enable errexit when EXECUTED, not when SOURCED — otherwise `set -e`
# leaks into your interactive shell and it can close on Tab-completion.
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0
[ "$SOURCED" = 0 ] && set -e

PYTHON=${PYTHON:-python3}
VENV_DIR="venv"

echo "==> Python: $($PYTHON --version)"

if [ -d "$VENV_DIR" ]; then
    echo "==> venv already exists — skipping creation"
else
    echo "==> Creating virtual environment in ./${VENV_DIR}/"
    $PYTHON -m venv "$VENV_DIR"
fi

# Use the venv's own pip directly (installs into ./venv, never the global Python).
VENV_PY="$VENV_DIR/bin/python"

# Safety check: make sure the venv actually has pip (missing python3-venv on
# some distros creates a venv without it).
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "ERROR: pip not found in the venv. On Debian/Ubuntu install it with:" >&2
    echo "       sudo apt install python3-venv python3-pip" >&2
    [ "$SOURCED" = 1 ] && return 1 || exit 1
fi

echo "==> Installing dependencies into $("$VENV_PY" -c 'import sys; print(sys.prefix)')"
"$VENV_PY" -m pip install --upgrade --quiet pip
"$VENV_PY" -m pip install --quiet -r requirements.txt

echo ""
echo "Done. Versions installed:"
"$VENV_PY" -m pip show torch opencv-python numpy | grep -E "^(Name|Version):"

echo ""
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Quick test:"
echo "  python predict.py --image images/sample/image_T1.jpg --save"
