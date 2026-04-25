import torch
from transformers import AutoTokenizer


class CMCNetCollator:
    def __init__(
        self,
        text_encoder_name: str = "microsoft/deberta-v3-base",
        max_length: int = 128,
        patch_grid: int = 14,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, use_fast=True)
        self.max_length = max_length
        self.patch_grid = patch_grid

    def __call__(self, batch: list[dict]) -> dict:
        texts = [sample["text"] for sample in batch]

        # Tokenize all texts at once
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mappings = encoding["offset_mapping"]

        images = []
        binary_labels = []
        type_labels = []
        text_grounding_labels = []
        image_grounding_labels = []
        has_text_grounding = []
        has_image_grounding = []

        for i, sample in enumerate(batch):
            images.append(sample["image"])
            binary_labels.append(sample["binary_label"])
            type_labels.append(sample["type_label"])

            # Text grounding labels
            tg_labels = self._build_text_grounding_labels(
                sample["text"],
                sample["fake_text_pos"],
                offset_mappings[i],
                sample["has_text_grounding"],
            )
            text_grounding_labels.append(tg_labels)

            # Image grounding labels
            ig_labels = self._build_image_grounding_labels(sample["fake_image_box"])
            image_grounding_labels.append(ig_labels)

            has_text_grounding.append(sample["has_text_grounding"])
            has_image_grounding.append(sample["has_image_grounding"])

        return {
            "images": torch.stack(images),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "binary_labels": torch.tensor(binary_labels, dtype=torch.long),
            "type_labels": torch.tensor(type_labels, dtype=torch.float32),
            "text_grounding_labels": torch.stack(text_grounding_labels),
            "image_grounding_labels": torch.stack(image_grounding_labels),
            "has_text_grounding": torch.tensor(has_text_grounding, dtype=torch.bool),
            "has_image_grounding": torch.tensor(has_image_grounding, dtype=torch.bool),
        }

    def _build_text_grounding_labels(
        self,
        text: str,
        fake_text_pos: list,
        offset_mapping: torch.Tensor,
        has_manipulation: bool,
    ) -> torch.Tensor:
        """Build per-subtoken grounding labels using char-level alignment."""
        labels = torch.full((self.max_length,), -100.0)

        # Mark all content tokens as 0 (real) first
        for idx in range(self.max_length):
            start, end = int(offset_mapping[idx][0]), int(offset_mapping[idx][1])
            if start == 0 and end == 0:
                continue
            labels[idx] = 0.0

        if not has_manipulation or not fake_text_pos:
            return labels

 
        words = text.split()
        word_char_spans = []
        char_pos = 0
        for word in words:
            while char_pos < len(text) and text[char_pos] == " ":
                char_pos += 1
            word_char_spans.append((char_pos, char_pos + len(word)))
            char_pos += len(word)

        # Build fake character set
        fake_char_set = set()
        for word_idx in fake_text_pos:
            if word_idx < len(word_char_spans):
                start, end = word_char_spans[word_idx]
                fake_char_set.update(range(start, end))

        # Map to subtokens: if any char of the subtoken is fake, mark as 1.0
        for idx in range(self.max_length):
            tok_start = int(offset_mapping[idx][0])
            tok_end = int(offset_mapping[idx][1])
            if tok_start == 0 and tok_end == 0:
                continue
            for c in range(tok_start, tok_end):
                if c in fake_char_set:
                    labels[idx] = 1.0
                    break

        return labels

    def _build_image_grounding_labels(
        self, bbox: list | None, image_size: int = 224
    ) -> torch.Tensor:
        """Build normalized xyxy bbox labels for image grounding."""
        labels = torch.zeros(4)

        if bbox is None:
            return labels

        x1, y1, x2, y2 = bbox
        labels[:] = torch.tensor([
            x1 / image_size,
            y1 / image_size,
            x2 / image_size,
            y2 / image_size,
        ])
        labels.clamp_(0.0, 1.0)

        return labels
