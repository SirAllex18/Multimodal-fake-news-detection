import torch.nn as nn


class BinaryHead(nn.Module):
    """Binary authenticity classification from fused CLS tokens."""
    def __init__(self, input_dim: int = 1536):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, fused_cls):
        return self.classifier(fused_cls)  # (B, 2)


class TypeHead(nn.Module):
    """9-class manipulation type classification."""
    def __init__(self, input_dim: int = 1536, num_classes: int = 9):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, fused_cls):
        return self.classifier(fused_cls)  # (B, num_classes)


class TextGroundingHead(nn.Module):
    """Per-token manipulation prediction."""
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, text_fused):
        return self.classifier(text_fused).squeeze(-1)  # (B, seq_len)


class ImageGroundingHead(nn.Module):
    """Per-patch manipulation prediction. CLS must be excluded by caller."""
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, image_patches_fused):
        return self.classifier(image_patches_fused).squeeze(-1)  # (B, num_patches)
