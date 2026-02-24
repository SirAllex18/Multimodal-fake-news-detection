import pytest
import torch

from src.losses.multi_task_loss import (
    focal_bce_with_logits,
    dice_loss,
    consistency_reg,
    MultiTaskLoss,
)


class TestFocalLoss:
    def test_shape(self):
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        loss = focal_bce_with_logits(logits, targets)
        assert loss.shape == (10,)

    def test_reduces_to_bce_at_gamma_zero(self):
        """With gamma=0 and alpha=0.5, focal loss equals standard BCE."""
        logits = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()

        focal = focal_bce_with_logits(logits, targets, gamma=0.0, alpha=0.5)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # alpha=0.5 means factor of 0.5 applied to both classes
        torch.testing.assert_close(focal, 0.5 * bce, atol=1e-5, rtol=1e-5)

    def test_mean_reduction(self):
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        loss = focal_bce_with_logits(logits, targets, reduction="mean")
        assert loss.dim() == 0  # scalar


class TestDiceLoss:
    def test_shape(self):
        logits = torch.randn(4, 196)
        targets = torch.randint(0, 2, (4, 196)).float()
        loss = dice_loss(logits, targets)
        assert loss.dim() == 0

    def test_perfect_prediction(self):
        """When prediction perfectly matches target, loss should be near 0."""
        targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        # Large positive logits where target=1, large negative where target=0
        logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
        loss = dice_loss(logits, targets)
        assert loss.item() < 0.1

    def test_all_zeros(self):
        """All-zero target and prediction should give loss near 0 (due to smoothing)."""
        logits = torch.full((1, 196), -10.0)
        targets = torch.zeros(1, 196)
        loss = dice_loss(logits, targets)
        assert loss.item() < 0.1


class TestConsistencyReg:
    def test_returns_zero_no_mixed(self):
        """No mixed samples -> loss is 0."""
        binary = torch.randn(4, 2)
        img_ground = torch.randn(4, 196)
        txt_ground = torch.randn(4, 128)
        has_img = torch.tensor([True, True, False, False])
        has_txt = torch.tensor([False, False, True, True])

        loss = consistency_reg(binary, img_ground, txt_ground, has_img, has_txt)
        assert loss.item() == 0.0

    def test_returns_nonzero_with_mixed(self):
        """Mixed samples should produce some loss."""
        binary = torch.tensor([[0.0, 5.0], [0.0, 5.0], [0.0, -5.0], [0.0, -5.0]])
        img_ground = torch.randn(4, 196) * 0.01  # Low grounding signal
        txt_ground = torch.randn(4, 128) * 0.01
        has_img = torch.tensor([True, True, False, False])
        has_txt = torch.tensor([True, False, True, False])

        loss = consistency_reg(binary, img_ground, txt_ground, has_img, has_txt)
        # First sample is mixed with high det_conf and low grounding
        assert loss.item() > 0.0


class TestMultiTaskLoss:
    @pytest.fixture
    def loss_fn(self):
        config = {
            "lambda_binary": 1.0,
            "lambda_type": 0.3,
            "lambda_text_grounding": 0.5,
            "lambda_image_grounding": 0.5,
            "lambda_consistency": 0.1,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "dice_weight": 0.3,
        }
        return MultiTaskLoss(config)

    def test_no_nan_all_orig_batch(self, loss_fn):
        """All-orig batch should produce no NaN."""
        outputs = {
            "binary_logits": torch.randn(4, 2),
            "type_logits": torch.randn(4, 9),
            "text_ground_logits": torch.randn(4, 128),
            "image_ground_logits": torch.randn(4, 196),
        }
        batch = {
            "binary_labels": torch.zeros(4, dtype=torch.long),
            "type_labels": torch.zeros(4, dtype=torch.long),
            "text_grounding_labels": torch.full((4, 128), -100.0),
            "image_grounding_labels": torch.zeros(4, 196),
            "has_text_grounding": torch.tensor([False, False, False, False]),
            "has_image_grounding": torch.tensor([False, False, False, False]),
        }

        losses = loss_fn(outputs, batch)
        assert not torch.isnan(losses["total"])
        assert losses["text_grounding"].item() == 0.0
        assert losses["image_grounding"].item() == 0.0

    def test_grounding_mask(self, loss_fn):
        """Grounding losses only computed for relevant samples."""
        outputs = {
            "binary_logits": torch.randn(4, 2),
            "type_logits": torch.randn(4, 9),
            "text_ground_logits": torch.randn(4, 128),
            "image_ground_logits": torch.randn(4, 196),
        }

        tg_labels = torch.full((4, 128), -100.0)
        # Only sample 1 has text grounding
        tg_labels[1, 1:10] = 0.0
        tg_labels[1, 3] = 1.0

        ig_labels = torch.zeros(4, 196)
        # Only sample 2 has image grounding
        ig_labels[2, 10:20] = 1.0

        batch = {
            "binary_labels": torch.tensor([0, 1, 1, 0]),
            "type_labels": torch.tensor([0, 3, 1, 0]),
            "text_grounding_labels": tg_labels,
            "image_grounding_labels": ig_labels,
            "has_text_grounding": torch.tensor([False, True, False, False]),
            "has_image_grounding": torch.tensor([False, False, True, False]),
        }

        losses = loss_fn(outputs, batch)
        assert not torch.isnan(losses["total"])
        assert losses["text_grounding"].item() > 0.0
        assert losses["image_grounding"].item() > 0.0

    def test_gradient_flow(self, loss_fn):
        """Verify gradients flow through the loss."""
        logits = torch.randn(2, 2, requires_grad=True)
        outputs = {
            "binary_logits": logits,
            "type_logits": torch.randn(2, 9, requires_grad=True),
            "text_ground_logits": torch.randn(2, 128, requires_grad=True),
            "image_ground_logits": torch.randn(2, 196, requires_grad=True),
        }
        batch = {
            "binary_labels": torch.tensor([0, 1]),
            "type_labels": torch.tensor([0, 1]),
            "text_grounding_labels": torch.zeros(2, 128),
            "image_grounding_labels": torch.zeros(2, 196),
            "has_text_grounding": torch.tensor([False, True]),
            "has_image_grounding": torch.tensor([False, True]),
        }

        losses = loss_fn(outputs, batch)
        losses["total"].backward()
        assert logits.grad is not None
