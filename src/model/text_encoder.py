import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """DeBERTa-v3-base wrapper returning full sequence hidden states."""

    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  # 768

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, seq_len, 768)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_top_n_layers(self, n: int):
        total = len(self.model.encoder.layer)
        for i in range(total - n, total):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True
        # Unfreeze relative position embeddings if present
        if hasattr(self.model.encoder, "rel_embeddings"):
            for param in self.model.encoder.rel_embeddings.parameters():
                param.requires_grad = True
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = True
