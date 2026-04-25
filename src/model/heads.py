import torch
import torch.nn as nn


class BinaryHead(nn.Module):
    """Binary authenticity classification from fused CLS tokens plus similarity."""
    def __init__(self, input_dim: int = 1537, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, fused_cls):
        return self.classifier(fused_cls)  # (B, 2)


class ProjectionHead(nn.Module):
    """Project modality-specific global features into a shared contrastive space."""
    def __init__(self, input_dim: int = 768, projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class TypeHead(nn.Module):
    """4-label manipulation type classification: FS, FA, TS, TA."""
    def __init__(self, input_dim: int = 1536, num_classes: int = 4,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_cls):
        return self.classifier(fused_cls)  # (B, num_classes)


class TextGroundingHead(nn.Module):
    """Per-token manipulation prediction."""
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_fused):
        return self.classifier(text_fused).squeeze(-1)  # (B, seq_len)


class ITMHead(nn.Module):
    """Image-text matching classification over fused pair representations."""
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, fused_cls: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused_cls)


class ImageGroundingHead(nn.Module):
    """Attention-pool spatial image tokens and regress normalized bbox logits."""
    def __init__(self, hidden_dim: int = 768, attn_hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, attn_hidden_dim),
            nn.GELU(),
            nn.Linear(attn_hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, image_patches_fused):
        attn_logits = self.attention(image_patches_fused).squeeze(-1)
        attn_weights = attn_logits.softmax(dim=-1).unsqueeze(-1)
        pooled = (image_patches_fused * attn_weights).sum(dim=1)
        return self.regressor(pooled)  # (B, 4)
