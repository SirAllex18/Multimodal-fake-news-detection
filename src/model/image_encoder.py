import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class ImageEncoder(nn.Module):
    """
    CLIP ViT-B/16 wrapper returning ALL patch tokens (not just CLS).

    Returns (B, 197, 768) = [CLS] + 196 patch tokens, pre-projection.
    Does NOT apply visual.proj (would reduce 768->512).
    """

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai"):
        super().__init__()
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = clip_model.visual
        self.hidden_size = 768

        # Delete unneeded text components
        del clip_model.transformer
        del clip_model.token_embedding
        del clip_model.ln_final

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.visual.conv1(images)                  # (B, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1)     # (B, 768, 196)
        x = x.permute(0, 2, 1)                        # (B, 196, 768)

        # Prepend CLS token
        cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)          # (B, 197, 768)

        # Positional embeddings (with interpolation for non-224 input sizes)
        pos_embed = self.visual.positional_embedding.to(x.dtype)
        if x.shape[1] != pos_embed.shape[0]:
            cls_pos = pos_embed[:1, :]    # (1, 768)
            patch_pos = pos_embed[1:, :]  # (N_orig, 768)
            grid_size = int(patch_pos.shape[0] ** 0.5)  # 14 for ViT-B/16
            new_grid = int((x.shape[1] - 1) ** 0.5)
            patch_pos = patch_pos.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_embed.shape[-1])
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        x = x + pos_embed
        x = self.visual.ln_pre(x)

        # Transformer (open_clip uses seq-first internally)
        x = x.permute(1, 0, 2)                        # (197, B, 768)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)                        # (B, 197, 768)

        # Final LayerNorm on ALL tokens
        x = self.visual.ln_post(x)

        # Do NOT apply self.visual.proj
        return x

    def freeze(self):
        for param in self.visual.parameters():
            param.requires_grad = False

    def unfreeze_top_n_layers(self, n: int):
        total = len(self.visual.transformer.resblocks)
        for i in range(total - n, total):
            for param in self.visual.transformer.resblocks[i].parameters():
                param.requires_grad = True
        for param in self.visual.ln_post.parameters():
            param.requires_grad = True
