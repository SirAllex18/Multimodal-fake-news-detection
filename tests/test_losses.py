import pytest
import torch

from src.losses.multi_task_loss import (
    focal_bce_with_logits,
    dice_loss,
    bbox_from_logits,
    generalized_box_iou,
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


class TestBoxLossHelpers:
    def test_bbox_from_logits_valid_boxes(self):
        logits = torch.randn(8, 4)
        boxes = bbox_from_logits(logits)
        assert boxes.shape == (8, 4)
        assert torch.all(boxes >= 0.0)
        assert torch.all(boxes <= 1.0)
        assert torch.all(boxes[:, 2] >= boxes[:, 0])
        assert torch.all(boxes[:, 3] >= boxes[:, 1])

    def test_bbox_from_logits_uses_cxcywh(self):
        logits = torch.zeros(1, 4)
        boxes = bbox_from_logits(logits)
        torch.testing.assert_close(
            boxes,
            torch.tensor([[0.25, 0.25, 0.75, 0.75]]),
        )

    def test_generalized_box_iou_identity(self):
        boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        giou = generalized_box_iou(boxes, boxes)
        torch.testing.assert_close(giou, torch.ones(1, 1))


class TestMultiTaskLoss:
    @pytest.fixture
    def loss_fn(self):
        config = {
            "lambda_binary": 1.0,
            "lambda_type": 0.3,
            "lambda_text_grounding": 0.5,
            "lambda_image_grounding": 0.5,
            "lambda_alignment": 0.1,
            "lambda_itm": 0.2,
            "lambda_consistency": 0.0,
            "contrastive_temp": 0.07,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "dice_weight": 0.3,
            "bbox_l1_weight": 5.0,
            "bbox_giou_weight": 2.0,
        }
        return MultiTaskLoss(config)

    def test_no_nan_all_orig_batch(self, loss_fn):
        """All-orig batch should produce no NaN."""
        outputs = {
            "binary_logits": torch.randn(4, 2),
            "type_logits": torch.randn(4, 4),
            "text_ground_logits": torch.randn(4, 128),
            "image_ground_logits": torch.randn(4, 4),
        }
        batch = {
            "binary_labels": torch.zeros(4, dtype=torch.long),
            "type_labels": torch.zeros(4, 4),
            "text_grounding_labels": torch.full((4, 128), -100.0),
            "image_grounding_labels": torch.zeros(4, 4),
            "has_text_grounding": torch.tensor([False, False, False, False]),
            "has_image_grounding": torch.tensor([False, False, False, False]),
        }

        losses = loss_fn(outputs, batch)
        assert not torch.isnan(losses["total"])
        assert losses["text_grounding"].item() == 0.0
        assert losses["image_grounding"].item() == 0.0
        assert losses["alignment"].item() == 0.0
        assert losses["itm"].item() == 0.0

    def test_grounding_mask(self, loss_fn):
        """Grounding losses only computed for relevant samples."""
        outputs = {
            "binary_logits": torch.randn(4, 2),
            "type_logits": torch.randn(4, 4),
            "text_ground_logits": torch.randn(4, 128),
            "image_ground_logits": torch.randn(4, 4),
        }

        tg_labels = torch.full((4, 128), -100.0)
        # Only sample 1 has text grounding
        tg_labels[1, 1:10] = 0.0
        tg_labels[1, 3] = 1.0

        ig_labels = torch.zeros(4, 4)
        # Only sample 2 has image grounding
        ig_labels[2] = torch.tensor([0.1, 0.1, 0.5, 0.5])

        batch = {
            "binary_labels": torch.tensor([0, 1, 1, 0]),
            "type_labels": torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
            "text_grounding_labels": tg_labels,
            "image_grounding_labels": ig_labels,
            "has_text_grounding": torch.tensor([False, True, False, False]),
            "has_image_grounding": torch.tensor([False, False, True, False]),
        }

        losses = loss_fn(outputs, batch)
        assert not torch.isnan(losses["total"])
        assert losses["text_grounding"].item() > 0.0
        assert losses["image_grounding"].item() > 0.0

    def test_stage_b_losses_use_orig_pairs_only(self, loss_fn):
        outputs = {
            "binary_logits": torch.randn(4, 2),
            "type_logits": torch.randn(4, 4),
            "text_ground_logits": torch.randn(4, 128),
            "image_ground_logits": torch.randn(4, 4),
            "text_proj": torch.nn.functional.normalize(torch.randn(4, 256), dim=-1),
            "image_proj": torch.nn.functional.normalize(torch.randn(4, 256), dim=-1),
            "itm_pos_logits": torch.randn(1, 2),
            "itm_neg_logits": torch.randn(2, 2),
        }
        batch = {
            "binary_labels": torch.tensor([0, 1, 1, 1]),
            "type_labels": torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ]),
            "text_grounding_labels": torch.full((4, 128), -100.0),
            "image_grounding_labels": torch.zeros(4, 4),
            "has_text_grounding": torch.tensor([False, False, False, False]),
            "has_image_grounding": torch.tensor([False, False, False, False]),
        }

        losses = loss_fn(outputs, batch)
        assert losses["alignment"].item() > 0.0
        assert losses["itm"].item() > 0.0

    def test_stage_b_queue_logits_path(self, loss_fn):
        outputs = {
            "binary_logits": torch.randn(2, 2),
            "type_logits": torch.randn(2, 4),
            "text_ground_logits": torch.randn(2, 128),
            "image_ground_logits": torch.randn(2, 4),
            "alignment_logits_t2i": torch.randn(2, 6),
            "alignment_logits_i2t": torch.randn(2, 6),
            "alignment_targets": torch.tensor([0, 1]),
            "itm_logits": torch.randn(4, 2),
            "itm_labels": torch.tensor([1, 1, 0, 0]),
        }
        batch = {
            "binary_labels": torch.tensor([0, 0]),
            "type_labels": torch.zeros(2, 4),
            "text_grounding_labels": torch.full((2, 128), -100.0),
            "image_grounding_labels": torch.zeros(2, 4),
            "has_text_grounding": torch.tensor([False, False]),
            "has_image_grounding": torch.tensor([False, False]),
        }

        losses = loss_fn(outputs, batch)
        assert losses["alignment"].item() > 0.0
        assert losses["itm"].item() > 0.0

    def test_stage_b_losses_skip_all_fake_batch(self, loss_fn):
        outputs = {
            "binary_logits": torch.randn(3, 2),
            "type_logits": torch.randn(3, 4),
            "text_ground_logits": torch.randn(3, 128),
            "image_ground_logits": torch.randn(3, 4),
            "text_proj": torch.nn.functional.normalize(torch.randn(3, 256), dim=-1),
            "image_proj": torch.nn.functional.normalize(torch.randn(3, 256), dim=-1),
        }
        batch = {
            "binary_labels": torch.tensor([1, 1, 1]),
            "type_labels": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ]),
            "text_grounding_labels": torch.full((3, 128), -100.0),
            "image_grounding_labels": torch.zeros(3, 4),
            "has_text_grounding": torch.tensor([False, False, False]),
            "has_image_grounding": torch.tensor([False, False, False]),
        }

        losses = loss_fn(outputs, batch)
        assert losses["alignment"].item() == 0.0
        assert losses["itm"].item() == 0.0

    def test_gradient_flow(self, loss_fn):
        """Verify gradients flow through the loss."""
        logits = torch.randn(2, 2, requires_grad=True)
        outputs = {
            "binary_logits": logits,
            "type_logits": torch.randn(2, 4, requires_grad=True),
            "text_ground_logits": torch.randn(2, 128, requires_grad=True),
            "image_ground_logits": torch.randn(2, 4, requires_grad=True),
            "text_proj": torch.nn.functional.normalize(torch.randn(2, 256, requires_grad=True), dim=-1),
            "image_proj": torch.nn.functional.normalize(torch.randn(2, 256, requires_grad=True), dim=-1),
            "itm_pos_logits": torch.randn(1, 2, requires_grad=True),
            "itm_neg_logits": torch.randn(2, 2, requires_grad=True),
        }
        batch = {
            "binary_labels": torch.tensor([0, 1]),
            "type_labels": torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            "text_grounding_labels": torch.zeros(2, 128),
            "image_grounding_labels": torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.5, 0.5],
            ]),
            "has_text_grounding": torch.tensor([False, True]),
            "has_image_grounding": torch.tensor([False, True]),
        }

        losses = loss_fn(outputs, batch)
        losses["total"].backward()
        assert logits.grad is not None
