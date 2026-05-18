## Setup

```bash
# CUDA 11.8 vorausgesetzt
pip install -r requirements.txt

# TensorFlow deinstallieren falls vorhanden
pip uninstall tensorflow -y

# Training
python train.py

# Evaluation
python evaluate.py
```

## Hardware
- Entwickelt auf: Tesla V100 16GB (QLoRA)
- Empfohlen für Full Fine-Tuning: A100 40GB+