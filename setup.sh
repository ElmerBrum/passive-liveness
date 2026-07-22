#!/usr/bin/env bash
# One-time environment setup for our-liveness.
# Run once from the our-liveness/ directory:
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

echo "==> Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade --quiet pip
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt

echo ""
echo "Done. Versions installed:"
"$VENV_DIR/bin/pip" show torch opencv-python numpy | grep -E "^(Name|Version):"

echo ""
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Quick test:"
echo "  python predict.py --image images/sample/image_T1.jpg --save"
