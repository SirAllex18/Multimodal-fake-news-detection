import pytest
import os

from src.data.dgm4_dataset import DGM4Dataset

# Resolve project root relative to this test file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DGM4_ROOT = os.path.join(PROJECT_ROOT, "DGM4")
HAS_DGM4 = os.path.isdir(DGM4_ROOT)


@pytest.fixture
def dataset_root():
    return DGM4_ROOT


def _make_minimal_dataset(dataset_root):
    """Create a dataset instance without loading annotations."""
    dataset = DGM4Dataset.__new__(DGM4Dataset)
    dataset.dataset_root = dataset_root
    return dataset


class TestImagePathResolution:
    """Test _resolve_image_path for various subdirectories."""

    @pytest.mark.skipif(not HAS_DGM4, reason="DGM4 dataset not available")
    def test_resolve_origin_path(self, dataset_root):
        """Origin images use nested structure: origin/usa_today/0048/120.jpg"""
        dataset = _make_minimal_dataset(dataset_root)
        test_path = "DGM4/origin/usa_today/0048/120.jpg"
        try:
            resolved = dataset._resolve_image_path(test_path)
            assert os.path.exists(resolved)
        except FileNotFoundError:
            pytest.skip("Specific test image not found")

    @pytest.mark.skipif(not HAS_DGM4, reason="DGM4 dataset not available")
    def test_resolve_manipulation_nested_path(self, dataset_root):
        """Manipulation images have zip nesting: infoswap/infoswap/file.jpg"""
        dataset = _make_minimal_dataset(dataset_root)
        # This path from JSON doesn't have the nested subdir;
        # _resolve_image_path should try the nested version automatically
        test_path = "DGM4/manipulation/infoswap/649778-081830-infoswap.jpg"
        try:
            resolved = dataset._resolve_image_path(test_path)
            assert os.path.exists(resolved)
        except FileNotFoundError:
            pytest.skip("Specific test image not found")

    @pytest.mark.skipif(not HAS_DGM4, reason="DGM4 dataset not available")
    def test_resolve_simswap_path(self, dataset_root):
        """Test simswap subdirectory resolution."""
        dataset = _make_minimal_dataset(dataset_root)
        test_path = "DGM4/manipulation/simswap/683133-013337-simswap.jpg"
        try:
            resolved = dataset._resolve_image_path(test_path)
            assert os.path.exists(resolved)
        except FileNotFoundError:
            pytest.skip("Specific test image not found")

    def test_resolve_raises_on_missing(self, dataset_root):
        """Missing file raises FileNotFoundError."""
        dataset = _make_minimal_dataset(dataset_root)
        with pytest.raises(FileNotFoundError):
            dataset._resolve_image_path("DGM4/nonexistent/fake.jpg")

    @pytest.mark.skipif(not HAS_DGM4, reason="DGM4 dataset not available")
    def test_dataset_loads_val_split(self, dataset_root):
        """Verify DGM4Dataset can load the val split and return a sample."""
        dataset = DGM4Dataset(
            dataset_root=dataset_root,
            annotation_file="metadata/val.json",
            image_size=224,
            is_train=False,
        )
        assert len(dataset) > 0
        sample = dataset[0]
        assert sample["image"].shape == (3, 224, 224)
        assert isinstance(sample["text"], str)
        assert sample["binary_label"] in (0, 1)
