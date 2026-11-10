"""In-Process-Zugriff auf Mammoth (qev.DecisionEngine) fuer run_mammoth_test.py.

Abgeleitet aus ~/test_mammoth.py. Quantisierung (quant bzw. Umgebungsvariable MAMMOTH_QUANT):
  auto (Standard): A100/H100 (sm_80+) -> BF16 ohne Quantisierung; GPU ohne natives BF16 (V100, sm_70) -> float16 ohne
                   Quantisierung, wenn die Gewichte hoechstens 65% des VRAM belegen (Elephant 9B: ja), sonst 4-bit NF4
                   (Mammoth 27B: ja)
  nf4:             immer 4-bit NF4 + float16 (auch auf A100/H100)
  none:            nie quantisieren (auf der V100 reichen 32 GB dafuer nicht -> OOM)

    decider = MammothDecider()            # laedt das Modell (1-3 min)
    decider.decide(state, questions)      # -> {"answers": {name: {"noul": p}}, "usage": {...}}

state:     str oder dict (z.B. {"question":..., "schema":..., "rules":...})
questions: {name: {"type": "noul", "instructions": "..."}}
Zu lange Requests (> ~16k Tokens) wirft die Engine als ValueError; hier wird nicht gekuerzt.

Modellordner: MAMMOTH_MODEL_DIR (Standard ~/mammoth-1). Funktioniert fuer jedes Modell mit der qev-Engine (Mammoth, Elephant).
"""
import json
import os
import threading
from pathlib import Path

import torch


def _install_quantization_patch():
    from transformers import BitsAndBytesConfig
    from transformers.modeling_utils import PreTrainedModel

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    orig = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _from_pretrained(cls, *args, **kw):
        kw["quantization_config"] = qcfg
        return orig(cls, *args, **kw)

    PreTrainedModel.from_pretrained = _from_pretrained

    # Sicherheitsnetz: autocast(bfloat16) -> float16
    base = torch.autocast

    class _FP16Autocast(base):
        def __init__(self, device_type, dtype=None, *a, **k):
            super().__init__(device_type, torch.float16 if dtype == torch.bfloat16 else dtype, *a, **k)

    torch.autocast = torch.amp.autocast = _FP16Autocast


class MammothDecider:
    def __init__(self, model_dir=None, device="cuda:0", quant=None):
        model_dir = str(Path(model_dir or os.environ.get("MAMMOTH_MODEL_DIR") or "~/mammoth-1").expanduser())
        # is_bf16_supported() ist auf der V100 wegen Emulation True -> native Unterstuetzung ueber sm_80+ pruefen
        native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        quant = (quant or os.environ.get("MAMMOTH_QUANT") or "auto").lower()
        if quant not in ("auto", "nf4", "none"):
            raise ValueError(f"quant muss auto, nf4 oder none sein, nicht {quant!r}")
        weights_gb = sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors")) / 1e9
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        use_nf4 = quant == "nf4" or (quant == "auto" and not native_bf16 and weights_gb > 0.65 * vram_gb)
        if use_nf4:
            _install_quantization_patch()
        elif not native_bf16:
            note = " (Gewichte belegen ueber 65% des VRAM, wahrscheinlich OOM)" if weights_gb > 0.65 * vram_gb else ""
            print(f"[mammoth] GPU ohne natives BF16: lade unquantisiert in float16{note}")
        from qev.engine import DecisionEngine  # nach dem Patch importieren
        from qev.schema import SystemOneRequest
        self._request_cls = SystemOneRequest
        self.engine = DecisionEngine(model_dir, device=device, dtype="bfloat16" if (native_bf16 and not use_nf4) else "float16")
        self.quantized = use_nf4
        print(f"[mammoth] Quantisierung: {'4-bit NF4' if use_nf4 else 'keine'} (quant={quant}, Gewichte {weights_gb:.1f} GB, VRAM {vram_gb:.1f} GB)")
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)  # fuer exaktes Chunking
        except Exception as e:
            print(f"[warn] Tokenizer nicht ladbar, Chunking schaetzt chars/2.0: {e}")
            self.tokenizer = None
        self._lock = threading.Lock()  # eine GPU: Requests serialisieren
        self._state_as_text = False

    def _make_request(self, state, questions):
        if isinstance(state, dict) and self._state_as_text:
            state = _state_to_text(state)
        try:
            return self._request_cls(state=state, questions=questions)
        except Exception:
            if isinstance(state, dict) and not self._state_as_text:
                self._state_as_text = True  # Schema akzeptiert nur Strings -> ab jetzt als Text
                return self._request_cls(state=_state_to_text(state), questions=questions)
            raise

    def decide(self, state, questions):
        req = self._make_request(state, questions)
        with self._lock:
            resp, _ = self.engine.decide(req)
            torch.cuda.synchronize()
        answers = {k: a.model_dump(exclude_none=True) for k, a in resp.answers.items()}
        usage = getattr(resp, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        return {"answers": answers, "usage": usage if isinstance(usage, dict) else {}}


def _state_to_text(state: dict) -> str:
    return "\n\n".join(f"{k}:\n{v}" if isinstance(v, str) else f"{k}:\n{json.dumps(v)}" for k, v in state.items())
