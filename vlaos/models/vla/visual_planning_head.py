import torch
from torch import nn
from torch.nn import functional as F
from vlaos.models.vla.language_planning_head import LanguagePlanningHead

class VisualPlanningHead(LanguagePlanningHead):
    def __init__(
        self,
        vocab_size: int,
        visual_planning_vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        llm_emb_dim: int = 768,
        num_heads: int = 8,
        planning_mode: str = "implicit"
    ):
        super().__init__(
            vocab_size=vocab_size+visual_planning_vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            llm_emb_dim=llm_emb_dim,
            num_heads=num_heads,
            planning_mode=planning_mode,
        )