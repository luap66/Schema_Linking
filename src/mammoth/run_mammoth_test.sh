#!/usr/bin/env bash
# Startet den Mammoth-Test auf dem ICC-GPU-Host. Prueft vorher, ob Mammoth installiert ist
# (venv, Modellgewichte, importierbarer MammothDecider) und fuehrt sonst setup_mammoth.sh aus.
#
#   bash src/mammoth/run_mammoth_test.sh                 # 30 Fragen, seed 42 (wie Jev-Test)
#   bash src/mammoth/run_mammoth_test.sh --sample 0      # alle Spider-Ent-Fragen
#   bash src/mammoth/run_mammoth_test.sh --probe         # nur Rohantwort des Deciders anzeigen
#   REINSTALL=1 bash src/mammoth/run_mammoth_test.sh     # Installation erzwingen
#
# Alle Argumente gehen unveraendert an run_mammoth_test.py (siehe --help dort).
#
# Umgebung:
#   MAMMOTH_HOME       Installationsordner (Standard: /mnt/data/$USER/mammoth)
#   MAMMOTH_VENV       optional, venv-Ordner (Standard: $MAMMOTH_HOME/venv)
#   MAMMOTH_MODEL_DIR  optional, Ordner der Modellgewichte
#   MAMMOTH_SETUP_DIR  Ordner mit setup_mammoth.sh + mammoth_decider.py (Standard: Ordner dieses Skripts)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SETUP_DIR="${MAMMOTH_SETUP_DIR:-$HERE}"
export MAMMOTH_HOME="${MAMMOTH_HOME:-/mnt/data/${USER}/mammoth}"
VENV_PY="${MAMMOTH_VENV:-$MAMMOTH_HOME/venv}/bin/python"

for f in setup_mammoth.sh mammoth_decider.py; do
    [ -f "$SETUP_DIR/$f" ] || { echo "FEHLER: $SETUP_DIR/$f fehlt (MAMMOTH_SETUP_DIR setzen oder Datei hierher kopieren)" >&2; exit 1; }
done

model_present() {
    if [ -n "${MAMMOTH_MODEL_DIR:-}" ]; then
        [ -n "$(find "$MAMMOTH_MODEL_DIR" -maxdepth 3 -name config.json -print -quit 2>/dev/null)" ]
    else
        [ -n "$(find "$MAMMOTH_HOME" \( -name venv -o -name site-packages \) -prune -o -name config.json -print -quit 2>/dev/null)" ]
    fi
}

is_installed() {
    [ -x "$VENV_PY" ] || { echo "-> venv fehlt ($VENV_PY)"; return 1; }
    model_present || { echo "-> Modellgewichte nicht gefunden"; return 1; }
    PYTHONPATH="$SETUP_DIR" "$VENV_PY" -c "import torch, transformers, mammoth_decider" 2>/dev/null \
        || { echo "-> torch/transformers/mammoth_decider nicht importierbar"; return 1; }
}

echo "Pruefe Mammoth-Installation in $MAMMOTH_HOME ..."
if [ "${REINSTALL:-0}" = "1" ] || ! is_installed; then
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
export PYTHONPATH="$SETUP_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_PY" -u "$HERE/run_mammoth_test.py" "$@"
