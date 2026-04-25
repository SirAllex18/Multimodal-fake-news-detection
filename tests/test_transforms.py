import pytest
import torch
from PIL import Image

from src.data.transforms import DGM4Transform


class TestDGM4Transform:
    @pytest.fixture
    def train_transform(self):
        return DGM4Transform(image_size=224, is_train=True)

    @pytest.fixture
    def val_transform(self):
        return DGM4Transform(image_size=224, is_train=False)

    def _make_image(self, w=300, h=400):
        return Image.new("RGB", (w, h), color=(128, 128, 128))

    def test_output_shape_no_bbox(self, val_transform):
        image = self._make_image()
        img_out, bbox_out = val_transform(image, bbox=None)
        assert img_out.shape == (3, 224, 224)
        assert bbox_out is None

    def test_output_shape_with_bbox(self, val_transform):
        image = self._make_image()
        bbox = [50, 50, 200, 300]
        img_out, bbox_out = val_transform(image, bbox)
        assert img_out.shape == (3, 224, 224)
        if bbox_out is not None:
            assert len(bbox_out) == 4
            assert bbox_out[2] > bbox_out[0]  # width > 0
            assert bbox_out[3] > bbox_out[1]  # height > 0
            assert bbox_out[0] == pytest.approx(56.0, abs=1.0)
            assert bbox_out[1] == pytest.approx(28.0, abs=1.0)
            assert bbox_out[2] == pytest.approx(140.0, abs=1.0)
            assert bbox_out[3] == pytest.approx(168.0, abs=1.0)

    def test_bbox_within_bounds(self, val_transform):
        """Transformed bbox should be within image bounds."""
        image = self._make_image()
        bbox = [10, 10, 290, 390]
        img_out, bbox_out = val_transform(image, bbox)
        if bbox_out is not None:
            assert bbox_out[0] >= 0
            assert bbox_out[1] >= 0
            assert bbox_out[2] <= 224
            assert bbox_out[3] <= 224

    def test_bbox_tracking_through_flip(self):
        """Verify bbox is correctly tracked through square-image resize."""
        transform = DGM4Transform(image_size=224, is_train=False)
        image = self._make_image(256, 256)
        bbox = [10, 10, 50, 50]
        img_out, bbox_out = transform(image, bbox)
        if bbox_out is not None:
            scale = 224 / 256
            assert bbox_out[0] == pytest.approx(10 * scale, abs=1.0)
            assert bbox_out[1] == pytest.approx(10 * scale, abs=1.0)
            assert bbox_out[2] == pytest.approx(50 * scale, abs=1.0)
            assert bbox_out[3] == pytest.approx(50 * scale, abs=1.0)

    def test_small_corner_bbox_survives_letterbox(self):
        """Small corner bboxes should survive letterbox preprocessing."""
        transform = DGM4Transform(image_size=224, is_train=False)
        image = self._make_image(400, 400)
        bbox = [0, 0, 5, 5]
        img_out, bbox_out = transform(image, bbox)
        assert img_out.shape == (3, 224, 224)
        assert bbox_out is not None
        assert bbox_out[2] > bbox_out[0]
        assert bbox_out[3] > bbox_out[1]

    def test_train_transform_augments(self, train_transform):
        """Train transform should still apply stochastic augmentation."""
        import numpy as np
        arr = np.zeros((400, 300, 3), dtype=np.uint8)
        for j in range(300):
            arr[:, j, :] = int(255 * j / 300)
        image = Image.fromarray(arr)
        bbox = [50, 50, 200, 300]

        results = []
        for _ in range(20):
            img_out, bbox_out = train_transform(image, bbox)
            left_mean = img_out[:, :, :112].mean().item()
            right_mean = img_out[:, :, 112:].mean().item()
            results.append(round(left_mean - right_mean, 4))

        assert len(set(results)) > 1, "Train transform should produce varied outputs"
