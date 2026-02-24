import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Single bidirectional cross-attention layer (pre-norm)."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Text -> Image direction
        self.norm_t2i_q = nn.LayerNorm(hidden_dim)
        self.norm_t2i_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2i = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_t2i_ffn = nn.LayerNorm(hidden_dim)
        self.ffn_t2i = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Image -> Text direction
        self.norm_i2t_q = nn.LayerNorm(hidden_dim)
        self.norm_i2t_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn_i2t = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_i2t_ffn = nn.LayerNorm(hidden_dim)
        self.ffn_i2t = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        text_key_padding_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Text attends to Image
        q = self.norm_t2i_q(text_feats)
        kv = self.norm_t2i_kv(image_feats)
        text_cross, _ = self.cross_attn_t2i(
            query=q, key=kv, value=kv, need_weights=False,
        )
        text_feats = text_feats + text_cross
        text_feats = text_feats + self.ffn_t2i(self.norm_t2i_ffn(text_feats))

        # Image attends to Text
        q = self.norm_i2t_q(image_feats)
        kv = self.norm_i2t_kv(text_feats)
        image_cross, _ = self.cross_attn_i2t(
            query=q, key=kv, value=kv,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        image_feats = image_feats + image_cross
        image_feats = image_feats + self.ffn_i2t(self.norm_i2t_ffn(image_feats))

        return text_feats, image_feats


class BidirectionalCrossAttention(nn.Module):
    """Stack of CrossAttentionLayer modules."""

    def __init__(
        self,
        num_layers: int = 2,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, text_feats, image_feats, text_key_padding_mask=None):
        for layer in self.layers:
            text_feats, image_feats = layer(
                text_feats, image_feats, text_key_padding_mask
            )
        return text_feats, image_feats
