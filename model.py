import torch
import torch.nn as nn


class ExSLModel(nn.Module):
    """Extractive Schema Linking model.

    Wraps a (LoRA-adapted) base LLM with a binary relevance head that predicts
    whether a candidate column is needed to answer a question.

    Architecture (as in the ExSL paper):
        For each candidate column marked with « ... », extract hidden states
        E_α (at «) and E_ω (at »), concatenate them, and apply a linear layer:
            logit = w_relevance @ [E_α ⊕ E_ω]
    """

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.w_relevance = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        open_pos: list[int],
        close_pos: list[int],
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [1, seq_len]
            attention_mask: [1, seq_len]
            open_pos:       token positions of « markers (one per candidate)
            close_pos:      token positions of » markers (one per candidate)

        Returns:
            logits: [n_candidates] — raw (pre-sigmoid) relevance scores
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask) # Enthält die Vektorrepräsentationen aller eingegeben Tokens über alle Layer hinweg
        hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size] - Durch index -1 letzte Layer

        pair_vectors = []
        for alpha, omega in zip(open_pos, close_pos):
            # Vektor an der « Position – enkodiert die Semantik aller 
            # vorangegangenen Tokens (Schema DDL + Frage + bisherige Kandidaten)
            e_alpha = hidden[0, alpha, :]
            
            # Vektor an der » Position – enkodiert zusätzlich den Tabellen- 
            # und Spaltennamen des Kandidaten selbst
            e_omega = hidden[0, omega, :]
            # Konkatiniert die Vektoren und fügt sie der Liste aller Candidates hinzu
            pair_vectors.append(torch.cat([e_alpha, e_omega], dim=-1)) 

        # Fallback falls Trunkation dafür sorgt das keine Kandidaten da sind.
        if not pair_vectors:
            return torch.zeros(0, device=input_ids.device)

        C = torch.stack(pair_vectors).float()  # [n_candidates, hidden*2], cast to fp32
        logits = self.w_relevance(C).squeeze(-1)  # [n_candidates] Klassifikationskopf verarbeitet alle Candidates und entfernt überflüssige Dimension.
        return logits
