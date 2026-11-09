#!/usr/bin/env bash
# Startet den Mammoth-Test auf dem ICC-GPU-Host. Prueft vorher, ob Mammoth installiert ist
# (venv, Modellgewichte, importierbare Engine) und fuehrt sonst setup_mammoth.sh aus (falls vorhanden).
#
#   bash run_mammoth_test.sh                 # 30 Fragen, seed 42 (wie Jev-Test)
#   bash run_mammoth_test.sh --sample 0      # alle Spider-Ent-Fragen
#   bash run_mammoth_test.sh --probe         # nur Rohantwort des Deciders anzeigen
#   REINSTALL=1 bash run_mammoth_test.sh     # Installation erzwingen
#
# Alle Argumente gehen unveraendert an run_mammoth_test.py (siehe --help dort).
#
# Umgebung (Standardwerte passen zu ~/mammoth-env + ~/mammoth-1):
#   MAMMOTH_VENV       venv-Ordner            (Standard: ~/mammoth-env, sonst $MAMMOTH_HOME/venv)
#   MAMMOTH_MODEL_DIR  Ordner der Gewichte    (Standard: ~/mammoth-1, sonst $MAMMOTH_HOME)
#   MAMMOTH_HOME       Installationsordner    (Standard: /mnt/data/$USER/mammoth)
#   MAMMOTH_SETUP_DIR  Ordner mit setup_mammoth.sh (nur fuer die Neuinstallation noetig)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SETUP_DIR="${MAMMOTH_SETUP_DIR:-$HERE}"
export MAMMOTH_HOME="${MAMMOTH_HOME:-/mnt/data/${USER}/mammoth}"
if [ -z "${MAMMOTH_VENV:-}" ]; then
    if [ -x "$HOME/mammoth-env/bin/python" ]; then MAMMOTH_VENV="$HOME/mammoth-env"; else MAMMOTH_VENV="$MAMMOTH_HOME/venv"; fi
fi
if [ -z "${MAMMOTH_MODEL_DIR:-}" ]; then
    if [ -f "$HOME/mammoth-1/config.json" ]; then MAMMOTH_MODEL_DIR="$HOME/mammoth-1"; else MAMMOTH_MODEL_DIR="$MAMMOTH_HOME"; fi
fi
export MAMMOTH_VENV MAMMOTH_MODEL_DIR
VENV_PY="$MAMMOTH_VENV/bin/python"

model_present() {
    [ -f "$MAMMOTH_MODEL_DIR/config.json" ] && ls "$MAMMOTH_MODEL_DIR"/*.safetensors >/dev/null 2>&1
}

is_installed() {
    [ -x "$VENV_PY" ] || { echo "-> venv fehlt ($VENV_PY)"; return 1; }
    model_present || { echo "-> Modellgewichte fehlen in $MAMMOTH_MODEL_DIR"; return 1; }
    "$VENV_PY" -c "import torch, transformers, qev.engine, qev.schema" 2>/dev/null \
        || { echo "-> torch/transformers/qev (OneJev-Runtime) nicht importierbar"; return 1; }
}

echo "Pruefe Mammoth-Installation (venv=$MAMMOTH_VENV, Modell=$MAMMOTH_MODEL_DIR) ..."
if [ "${REINSTALL:-0}" = "1" ] || ! is_installed; then
    [ -f "$SETUP_DIR/setup_mammoth.sh" ] || {
        echo "FEHLER: Mammoth ist nicht (vollstaendig) installiert und $SETUP_DIR/setup_mammoth.sh fehlt." >&2
        echo "        setup_mammoth.sh dorthin kopieren oder MAMMOTH_SETUP_DIR setzen." >&2
        exit 1
    }
    echo "Installiere Mammoth (setup_mammoth.sh) ..."
    mkdir -p "$MAMMOTH_HOME"
    bash "$SETUP_DIR/setup_mammoth.sh"
    is_installed || { echo "FEHLER: Installation unvollstaendig, siehe Ausgabe oben" >&2; exit 1; }
else
    echo "Mammoth ist installiert."
fi

# Zusatzpakete fuer den Test (torch/transformers bleiben unangetastet)
if ! "$VENV_PY" -c "import mo_sql_parsing" 2>/dev/null; then
    echo "Installiere mo-sql-parsing ..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$VENV_PY" mo-sql-parsing
    else
        "$VENV_PY" -m pip install mo-sql-parsing
    fi
fi

cd "$REPO_ROOT"
# kein PYTORCH_CUDA_ALLOC_CONF=expandable_segments (auf den vGPU-VMs "operation not supported")
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$HERE:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_PY" -u "$HERE/run_mammoth_test.py" "$@"
