import pytest
import torch

from src.model.text_encoder import TextEncoder
from src.model.image_encoder import ImageEncoder
from src.model.cross_attention import BidirectionalCrossAttention
from src.model.heads import BinaryHead, TypeHead, TextGroundingHead, ImageGroundingHead
from src.model.cmc_net import CMCNet


@pytest.fixture
def config():
    return {
        "model": {
            "text_encoder_name": "microsoft/deberta-v3-base",
            "image_encoder_name": "ViT-B-16",
            "image_encoder_pretrained": "openai",
            "hidden_dim": 768,
            "num_cross_attn_layers": 2,
            "num_cross_attn_heads": 8,
            "cross_attn_ffn_dim": 3072,
            "cross_attn_dropout": 0.1,
            "num_classes": 9,
            "num_patches": 196,
            "num_patches_with_cls": 197,
            "patch_grid": 14,
        },
        "loss": {
            "lambda_binary": 1.0,
            "lambda_type": 0.3,
            "lambda_text_grounding": 0.5,
            "lambda_image_grounding": 0.5,
            "lambda_consistency": 0.1,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "dice_weight": 0.3,
        },
        "training": {
            "freeze_encoders": True,
            "lr": 1e-4,
            "max_epochs": 10,
            "batch_size": 2,
            "precision": "bf16-mixed",
            "gradient_clip_val": 1.0,
        },
    }


class TestCrossAttention:
    def test_forward_shapes(self):
        """Test cross-attention output shapes."""
        ca = BidirectionalCrossAttention(
            num_layers=2, hidden_dim=768, num_heads=8, ffn_dim=3072
        )
        text = torch.randn(2, 128, 768)
        image = torch.randn(2, 197, 768)
        mask = torch.zeros(2, 128, dtype=torch.bool)
        mask[:, 100:] = True  # Mask padding

        text_out, image_out = ca(text, image, mask)
        assert text_out.shape == (2, 128, 768)
        assert image_out.shape == (2, 197, 768)


class TestHeads:
    def test_binary_head(self):
        head = BinaryHead(1536)
        x = torch.randn(4, 1536)
        out = head(x)
        assert out.shape == (4, 2)

    def test_type_head(self):
        head = TypeHead(1536, 9)
        x = torch.randn(4, 1536)
        out = head(x)
        assert out.shape == (4, 9)

    def test_text_grounding_head(self):
        head = TextGroundingHead(768)
        x = torch.randn(4, 128, 768)
        out = head(x)
        assert out.shape == (4, 128)

    def test_image_grounding_head(self):
        head = ImageGroundingHead(768)
        x = torch.randn(4, 196, 768)
        out = head(x)
        assert out.shape == (4, 196)


class TestCMCNet:
    @pytest.mark.slow
    def test_forward_shapes(self, config):
        """Test full model forward pass output shapes."""
        model = CMCNet(config)
        model.eval()

        batch = {
            "images": torch.randn(2, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128, dtype=torch.long),
            "binary_labels": torch.tensor([0, 1]),
            "type_labels": torch.tensor([0, 1]),
            "text_grounding_labels": torch.zeros(2, 128),
            "image_grounding_labels": torch.zeros(2, 196),
            "has_text_grounding": torch.tensor([False, True]),
            "has_image_grounding": torch.tensor([False, True]),
        }

        with torch.no_grad():
            outputs = model(batch)

        assert outputs["binary_logits"].shape == (2, 2)
        assert outputs["type_logits"].shape == (2, 9)
        assert outputs["text_ground_logits"].shape == (2, 128)
        assert outputs["image_ground_logits"].shape == (2, 196)

    @pytest.mark.slow
    def test_freeze_param_counts(self, config):
        """Verify frozen encoder params are not trainable."""
        model = CMCNet(config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        # With frozen encoders, only cross-attention + heads should be trainable
        # Cross-attention ~19M + heads ~2M ≈ 21M
        assert trainable < total, "Some params should be frozen"
        assert trainable > 1_000_000, "Should have at least 1M trainable params"

        # Encoders should be frozen
        for param in model.text_encoder.parameters():
            assert not param.requires_grad
        for param in model.image_encoder.parameters():
            assert not param.requires_grad

    @pytest.mark.slow
    def test_unfreeze_top_layers(self, config):
        """Test phase 2 unfreezing."""
        # Test gradual_unfreeze=False (immediate unfreeze, legacy behavior)
        config_phase2 = dict(config)
        config_phase2["training"] = {
            "freeze_encoders": False,
            "unfreeze_top_n_layers": 4,
            "gradual_unfreeze": False,
            "encoder_lr": 5e-6,
            "fusion_lr": 3e-5,
            "head_lr": 5e-5,
            "max_epochs": 15,
            "batch_size": 8,
        }

        model = CMCNet(config_phase2)

        text_trainable = sum(
            p.numel() for p in model.text_encoder.parameters() if p.requires_grad
        )
        image_trainable = sum(
            p.numel() for p in model.image_encoder.parameters() if p.requires_grad
        )

        assert text_trainable > 0, "Top text encoder layers should be unfrozen"
        assert image_trainable > 0, "Top image encoder layers should be unfrozen"

        # Test gradual_unfreeze=True (default) — starts frozen at init
        config_gradual = dict(config)
        config_gradual["training"] = {
            "freeze_encoders": False,
            "unfreeze_top_n_layers": 4,
            "encoder_lr": 5e-6,
            "fusion_lr": 3e-5,
            "head_lr": 5e-5,
            "max_epochs": 15,
            "batch_size": 8,
        }

        model_gradual = CMCNet(config_gradual)

        text_trainable_gradual = sum(
            p.numel() for p in model_gradual.text_encoder.parameters() if p.requires_grad
        )
        assert text_trainable_gradual == 0, "Gradual unfreeze should start with frozen encoders"
        assert model_gradual._gradual_unfreeze is True
