## Setup

```bash
# CUDA 11.8 vorausgesetzt
pip install -r requirements.txt

# Projekt als editable Package installieren, damit "src...."-Imports und
# Datenpfade unabhaengig vom Arbeitsverzeichnis funktionieren
pip install -e .

# TensorFlow deinstallieren falls vorhanden
pip uninstall tensorflow -y

# Training
python src/QLora/train.py

# Evaluation
python src/QLora/evaluate.py
```

## Hardware
- Entwickelt auf: Tesla V100 16GB (QLoRA)
- Empfohlen für Full Fine-Tuning: A100 40GB+