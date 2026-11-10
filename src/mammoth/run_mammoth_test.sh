#!/usr/bin/env bash
# Startet den Mammoth-Test auf dem ICC-GPU-Host. Prueft vorher, ob Mammoth installiert ist
# (venv, Modellgewichte, importierbare Engine) und installiert sonst selbst: venv anlegen, Modell holen, Runtime via setup_mammoth.sh oder $MODELL/setup.sh.
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
#   MAMMOTH_MODEL_SOURCE  fehlt das Modell: von hier per rsync holen (z.B. user@gpu-v100s-01:~/mammoth-1)
#   MAMMOTH_MODEL_REPO    fehlt das Modell: von diesem Hugging-Face-Repo laden (Standard: baselabs/mammoth-1)
#   MAMMOTH_QUANT      auto (Standard) | nf4 | none  (oder --quant am Skript)
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
export MAMMOTH_MODEL_REPO="${MAMMOTH_MODEL_REPO:-baselabs/mammoth-1}"
VENV_PY="$MAMMOTH_VENV/bin/python"

model_present() {
    [ -f "$MAMMOTH_MODEL_DIR/config.json" ] && ls "$MAMMOTH_MODEL_DIR"/*.safetensors >/dev/null 2>&1
}

runtime_ok() {
    [ -x "$VENV_PY" ] || { echo "-> venv fehlt ($VENV_PY)"; return 1; }
    "$VENV_PY" -c "import torch, transformers, qev.engine, qev.schema" 2>/dev/null         || { echo "-> torch/transformers/qev (OneJev-Runtime) nicht importierbar"; return 1; }
}

ensure_venv() {
    [ -x "$VENV_PY" ] && return 0
    echo "Lege venv an: $MAMMOTH_VENV ..."
    mkdir -p "$(dirname "$MAMMOTH_VENV")"
    if command -v uv >/dev/null 2>&1; then
        uv venv "$MAMMOTH_VENV"
    else
        python3 -m venv "$MAMMOTH_VENV" || { echo "FEHLER: venv konnte nicht angelegt werden (python3-venv fehlt?)" >&2; exit 1; }
    fi
}

pip_install() {
    if command -v uv >/dev/null 2>&1; then uv pip install --python "$VENV_PY" "$@"; else "$VENV_PY" -m pip install "$@"; fi
}

# Runtime installieren: 1) setup_mammoth.sh (falls vorhanden), 2) offizielles setup.sh im Modellordner
# (installiert die OneJev-Runtime in die aktive Umgebung, daher mit aktiviertem venv).
run_setup() {
    mkdir -p "$MAMMOTH_HOME"
    if [ -f "$SETUP_DIR/setup_mammoth.sh" ]; then
        echo "Fuehre setup_mammoth.sh aus ..."
        bash "$SETUP_DIR/setup_mammoth.sh"
    elif [ -f "$MAMMOTH_MODEL_DIR/setup.sh" ]; then
        ensure_venv
        # setup.sh erwartet das Image pytorch/pytorch:2.8.0-cuda12.8 (torch + torchvision schon da) und behaelt dessen Build.
        if ! "$VENV_PY" -c "import torch, torchvision" 2>/dev/null; then
            echo "Installiere torch 2.8.0 + torchvision 0.23.0 (CUDA 12.8) ..."
            pip_install torch==2.8.0 torchvision==0.23.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
        fi
        command -v git >/dev/null || { echo "FEHLER: git fehlt (setup.sh klont OneJev und decision-index)" >&2; exit 1; }
        # setup.sh abgeaendert: causal-conv1d ist optional (braucht nvcc), Fehler dort nicht fatal; pip fehlt im uv-venv
        local ws="$MAMMOTH_HOME/runtime"; mkdir -p "$ws"
        local patched="$MAMMOTH_MODEL_DIR/.setup_run.sh"
        sed -e 's|^\(pip install .*causal-conv1d\)$|\1 \|\| echo "WARNUNG: causal-conv1d nicht installiert (PyTorch-Fallback)"|' \
            -e 's|^B=\$(cd "\$(dirname "\$0")" && pwd)|B="'"$MAMMOTH_MODEL_DIR"'"|' \
            "$MAMMOTH_MODEL_DIR/setup.sh" > "$patched"
        "$VENV_PY" -m pip --version >/dev/null 2>&1 || "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || pip_install pip
        echo "Fuehre setup.sh im venv $MAMMOTH_VENV aus (Arbeitsordner $ws) ..."
        # /mnt/data gehoert laut git einem anderen Nutzer ("dubious ownership"): safe.directory nur fuer diesen Aufruf, ohne globale Config
        ( export VIRTUAL_ENV="$MAMMOTH_VENV" PATH="$MAMMOTH_VENV/bin:$PATH" GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="*"; unset PYTHONHOME; bash "$patched" "$ws" )
        rm -f "$patched"
    else
        echo "FEHLER: $1" >&2
        echo "        Weder $SETUP_DIR/setup_mammoth.sh noch $MAMMOTH_MODEL_DIR/setup.sh vorhanden." >&2
        exit 1
    fi
    SETUP_DONE=1
}

# Modell holen: 1) MAMMOTH_MODEL_SOURCE (rsync-Quelle, z.B. user@host:~/mammoth-1)
#               2) MAMMOTH_MODEL_REPO (Hugging-Face-Repo, optional mit HF_TOKEN)
#               3) setup_mammoth.sh
download_model() {
    mkdir -p "$MAMMOTH_MODEL_DIR"
    if [ -n "${MAMMOTH_MODEL_SOURCE:-}" ]; then
        echo "Kopiere Modell von $MAMMOTH_MODEL_SOURCE nach $MAMMOTH_MODEL_DIR ..."
        rsync -a --partial --info=progress2 "${MAMMOTH_MODEL_SOURCE%/}/" "$MAMMOTH_MODEL_DIR/"
    elif [ -n "${MAMMOTH_MODEL_REPO:-}" ]; then
        ensure_venv
        "$VENV_PY" -c "import huggingface_hub" 2>/dev/null || pip_install huggingface_hub
        echo "Lade Modell $MAMMOTH_MODEL_REPO nach $MAMMOTH_MODEL_DIR ..."
        "$VENV_PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$MAMMOTH_MODEL_REPO', local_dir='$MAMMOTH_MODEL_DIR')"
    else
        if [ -f "$SETUP_DIR/setup_mammoth.sh" ]; then
            run_setup "Modellgewichte fehlen"
        else
            echo "FEHLER: Modellgewichte fehlen in $MAMMOTH_MODEL_DIR." >&2
            echo "        MAMMOTH_MODEL_SOURCE=user@host:~/mammoth-1 (rsync) oder MAMMOTH_MODEL_REPO=<hf-repo> setzen." >&2
            exit 1
        fi
    fi
}

SETUP_DONE=0
echo "Pruefe Mammoth-Installation (venv=$MAMMOTH_VENV, Modell=$MAMMOTH_MODEL_DIR) ..."
if ! model_present; then
    echo "-> Modellgewichte fehlen in $MAMMOTH_MODEL_DIR"
    download_model
    model_present || { echo "FEHLER: Modellgewichte nach dem Download nicht vollstaendig vorhanden" >&2; exit 1; }
fi
if [ "${REINSTALL:-0}" = "1" ] || ! runtime_ok; then
    run_setup "Runtime (venv/OneJev) fehlt"
    runtime_ok || { echo "FEHLER: Runtime nach dem Setup nicht nutzbar, siehe Ausgabe oben" >&2; exit 1; }
fi
echo "Mammoth ist installiert."

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
