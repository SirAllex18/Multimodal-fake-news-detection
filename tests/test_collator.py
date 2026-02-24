import pytest
import torch
from PIL import Image

from src.data.collator import CMCNetCollator

ENCODER_NAME = "microsoft/deberta-v3-base"


@pytest.fixture
def collator():
    return CMCNetCollator(
        text_encoder_name=ENCODER_NAME,
        max_length=128,
        patch_grid=14,
    )


def _make_sample(text, fake_text_pos=None, bbox=None, fake_cls="orig"):
    """Helper to create a minimal sample dict."""
    from src.data.label_utils import get_binary_label, get_type_label, has_image_manipulation, has_text_manipulation

    has_img = has_image_manipulation(fake_cls) and bbox is not None
    has_txt = has_text_manipulation(fake_cls)

    # Create a dummy 224x224 image tensor
    image = torch.randn(3, 224, 224)

    return {
        "image": image,
        "text": text,
        "binary_label": get_binary_label(fake_cls),
        "type_label": get_type_label(fake_cls),
        "fake_cls": fake_cls,
        "fake_image_box": bbox,
        "fake_text_pos": fake_text_pos or [],
        "has_image_grounding": has_img,
        "has_text_grounding": has_txt,
    }


class TestTextGroundingLabels:
    def test_word_to_subtoken_simple(self, collator):
        """Single fake word, likely no subword splits."""
        text = "The cat sat on the mat"
        # "cat" is word index 1
        sample = _make_sample(text, fake_text_pos=[1], fake_cls="text_swap")
        batch = collator([sample])

        labels = batch["text_grounding_labels"][0]

        # Verify at least one token is marked fake and most are real
        content_mask = labels != -100.0
        fake_count = (labels[content_mask] == 1.0).sum().item()
        real_count = (labels[content_mask] == 0.0).sum().item()
        assert fake_count >= 1, "Should have at least one fake token for 'cat'"
        assert real_count > fake_count, "Most tokens should be real"

    def test_word_to_subtoken_with_split(self, collator):
        """Word that gets split into multiple subtokens."""
        text = "The unprecedented event occurred"
        # "unprecedented" is word index 1
        sample = _make_sample(text, fake_text_pos=[1], fake_cls="text_swap")
        batch = collator([sample])

        labels = batch["text_grounding_labels"][0]

        # Count fake tokens — "unprecedented" should produce multiple subtokens
        content_mask = labels != -100.0
        fake_count = (labels[content_mask] == 1.0).sum().item()
        assert fake_count >= 1, "Should have at least one fake token for 'unprecedented'"

        # Verify real tokens exist too
        real_count = (labels[content_mask] == 0.0).sum().item()
        assert real_count > 0, "Other words should be labeled real"

    def test_word_to_subtoken_multiple_fakes(self, collator):
        """Multiple fake words."""
        text = "The quick brown fox jumped over the lazy dog"
        # "quick" (1) and "lazy" (7) are fake
        sample = _make_sample(text, fake_text_pos=[1, 7], fake_cls="text_swap")
        batch = collator([sample])

        labels = batch["text_grounding_labels"][0]
        content_mask = labels != -100.0

        # Should have multiple fake tokens (for both "quick" and "lazy")
        fake_count = (labels[content_mask] == 1.0).sum().item()
        assert fake_count >= 2, "Should have fake tokens for both 'quick' and 'lazy'"

        real_count = (labels[content_mask] == 0.0).sum().item()
        assert real_count > fake_count, "Most tokens should be real"

    def test_word_to_subtoken_empty(self, collator):
        """No fake words -> all content tokens labeled 0."""
        text = "This is a normal sentence"
        sample = _make_sample(text, fake_text_pos=[], fake_cls="orig")
        batch = collator([sample])

        labels = batch["text_grounding_labels"][0]
        for idx in range(128):
            assert labels[idx] in (-100.0, 0.0), f"Token {idx} should be -100 or 0"

    def test_word_to_subtoken_repeated_word(self, collator):
        """Same word appears twice, only one instance is fake."""
        text = "The dog chased the dog"
        # Second "dog" is word index 4
        sample = _make_sample(text, fake_text_pos=[4], fake_cls="text_swap")
        batch = collator([sample])

        labels = batch["text_grounding_labels"][0]
        encoding = collator.tokenizer(text, return_offsets_mapping=True, max_length=128, truncation=True)
        offsets = encoding["offset_mapping"]

        # Find positions of both "dog"s
        words = text.split()
        char_pos = 0
        word_spans = []
        for word in words:
            while char_pos < len(text) and text[char_pos] == " ":
                char_pos += 1
            word_spans.append((char_pos, char_pos + len(word)))
            char_pos += len(word)

        # First "dog" (index 1) should be real
        first_dog_start, first_dog_end = word_spans[1]
        # Second "dog" (index 4) should be fake
        second_dog_start, second_dog_end = word_spans[4]

        for idx, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s >= first_dog_start and s < first_dog_end:
                assert labels[idx] == 0.0, "First 'dog' should be real"
            if s >= second_dog_start and s < second_dog_end:
                assert labels[idx] == 1.0, "Second 'dog' should be fake"

    def test_special_tokens_masked(self, collator):
        """CLS/SEP/PAD get label -100."""
        text = "Hello world"
        sample = _make_sample(text, fake_text_pos=[], fake_cls="orig")
        batch = collator([sample])

        labels = batch["text_grounding_labels"][0]
        encoding = collator.tokenizer(text, return_offsets_mapping=True,
                                       max_length=128, truncation=True, padding="max_length")
        offsets = encoding["offset_mapping"]

        for idx, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                assert labels[idx] == -100.0, f"Special token at {idx} should be -100"

    def test_hyphens(self, collator):
        """Hyphenated word handling."""
        text = "The well-known scientist discovered something"
        sample = _make_sample(text, fake_text_pos=[1], fake_cls="text_swap")
        batch = collator([sample])
        labels = batch["text_grounding_labels"][0]

        # "well-known" is word index 1 — should have fake tokens
        fake_tokens = (labels == 1.0).sum().item()
        assert fake_tokens >= 1, "Hyphenated word should produce at least one fake token"

    def test_apostrophes(self, collator):
        """Apostrophe handling."""
        text = "It's a beautiful day today"
        sample = _make_sample(text, fake_text_pos=[0], fake_cls="text_swap")
        batch = collator([sample])
        labels = batch["text_grounding_labels"][0]

        fake_tokens = (labels == 1.0).sum().item()
        assert fake_tokens >= 1, "Apostrophe word should produce at least one fake token"


class TestImageGroundingLabels:
    def test_patch_grid_basic(self, collator):
        """Known bbox covers specific patches."""
        # Bbox covering top-left 32x32 pixels = 2x2 patches at 14x14 grid (16px patches)
        bbox = [0, 0, 32, 32]
        labels = collator._build_image_grounding_labels(bbox)
        assert labels.shape == (196,)
        # Center-inside: patch centers at (8, 8), (24, 8), (8, 24), (24, 24)
        # All within [0,32] x [0,32]
        assert labels[0] == 1.0  # (row=0, col=0) center=(8,8)
        assert labels[1] == 1.0  # (row=0, col=1) center=(24,8)
        assert labels[14] == 1.0  # (row=1, col=0) center=(8,24)
        assert labels[15] == 1.0  # (row=1, col=1) center=(24,24)

    def test_patch_grid_spanning(self, collator):
        """Bbox spanning a larger area."""
        bbox = [50, 50, 150, 150]
        labels = collator._build_image_grounding_labels(bbox)
        assert labels.shape == (196,)
        positive_count = labels.sum().item()
        assert positive_count > 0
        # Should cover roughly (100/224 * 14)^2 ≈ 6x6 = 36 patches
        assert positive_count > 10

    def test_patch_grid_none(self, collator):
        """No bbox -> all zeros."""
        labels = collator._build_image_grounding_labels(None)
        assert labels.shape == (196,)
        assert labels.sum().item() == 0.0


class TestBatchCollation:
    def test_full_batch_shapes(self, collator):
        """Verify all output tensor shapes for a collated batch."""
        samples = [
            _make_sample("Sample one text here", fake_text_pos=[2], bbox=[10, 10, 100, 100], fake_cls="face_swap&text_swap"),
            _make_sample("Sample two text here", fake_text_pos=[], bbox=None, fake_cls="orig"),
            _make_sample("Sample three text here", fake_text_pos=[1], bbox=None, fake_cls="text_attribute"),
            _make_sample("Sample four text here", fake_text_pos=[], bbox=[50, 50, 200, 200], fake_cls="face_attribute"),
        ]
        batch = collator(samples)

        assert batch["images"].shape == (4, 3, 224, 224)
        assert batch["input_ids"].shape == (4, 128)
        assert batch["attention_mask"].shape == (4, 128)
        assert batch["binary_labels"].shape == (4,)
        assert batch["type_labels"].shape == (4,)
        assert batch["text_grounding_labels"].shape == (4, 128)
        assert batch["image_grounding_labels"].shape == (4, 196)
        assert batch["has_text_grounding"].shape == (4,)
        assert batch["has_image_grounding"].shape == (4,)

        # Check dtypes
        assert batch["binary_labels"].dtype == torch.long
        assert batch["type_labels"].dtype == torch.long
        assert batch["text_grounding_labels"].dtype == torch.float32
        assert batch["image_grounding_labels"].dtype == torch.float32
        assert batch["has_text_grounding"].dtype == torch.bool
        assert batch["has_image_grounding"].dtype == torch.bool
