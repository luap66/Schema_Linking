#!/usr/bin/env bash
# Derselbe Test wie run_mammoth_test.sh, aber mit Elephant-1 (baselabs/elephant-1, 9B).
# Modell wird bei Bedarf von Hugging Face geladen (MIT-Lizenz, laut Modellkarte nicht gesperrt).
# Nutzt dieselbe OneJev-Runtime/dasselbe venv wie Mammoth.
#
#   bash run_elephant_test.sh --probe
#   bash run_elephant_test.sh                  # 30 Fragen, seed 42
#   bash run_elephant_test.sh --sample 0 --resume
#
# ELEPHANT_DIR ueberschreibt den Modellordner (Standard ~/elephant-1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAMMOTH_MODEL_DIR="${ELEPHANT_DIR:-$HOME/elephant-1}"
export MAMMOTH_MODEL_REPO="${MAMMOTH_MODEL_REPO:-baselabs/elephant-1}"
exec bash "$HERE/run_mammoth_test.sh" --model-name elephant "$@"
