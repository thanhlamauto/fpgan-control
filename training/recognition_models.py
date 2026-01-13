#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
Recognition models for fingerprint verification.
Modified ResNet-18 with CosFace loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional, Tuple
import math


class ModifiedResNet18(nn.Module):
    """
    Modified ResNet-18 for fingerprint recognition.

    Key modifications per paper:
    - Remove first max pooling layer (avoid early downsampling)
    - Output embedding instead of classification
    """

    def __init__(
        self,
        embedding_size: int = 512,
        dropout: float = 0.0,
        remove_first_maxpool: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()

        # Load base ResNet-18
        if pretrained:
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base_model = resnet18(weights=None)

        # First conv layer
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        # Optionally remove maxpool (key modification per paper)
        self.remove_maxpool = remove_first_maxpool
        if not remove_first_maxpool:
            self.maxpool = base_model.maxpool

        # Residual layers
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Global average pooling
        self.avgpool = base_model.avgpool

        # Embedding layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(512, embedding_size)

        # Batch normalization for embedding (helps with CosFace)
        self.bn_embedding = nn.BatchNorm1d(embedding_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding layer weights."""
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out', nonlinearity='relu')
        if self.embedding.bias is not None:
            nn.init.constant_(self.embedding.bias, 0)
        nn.init.constant_(self.bn_embedding.weight, 1)
        nn.init.constant_(self.bn_embedding.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            embeddings: Feature embeddings [B, embedding_size]
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Skip maxpool if configured (important for fingerprints!)
        if not self.remove_maxpool:
            x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Embedding
        x = self.dropout(x)
        x = self.embedding(x)
        x = self.bn_embedding(x)

        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward for clarity during inference."""
        return self.forward(x)


class CosFaceLoss(nn.Module):
    """
    CosFace (Large Margin Cosine Loss) for face/fingerprint recognition.

    Reference: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.09414

    L = -log(exp(s*(cos(theta_yi) - m)) / (exp(s*(cos(theta_yi) - m)) + sum_j!=yi(exp(s*cos(theta_j)))))
    """

    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 50000,
        scale: float = 64.0,
        margin: float = 0.35,
    ):
        """
        Args:
            embedding_size: Size of input embeddings
            num_classes: Number of identity classes
            scale: Scaling factor s
            margin: Cosine margin m
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Learnable weight matrix (class centers)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CosFace loss.

        Args:
            embeddings: Feature embeddings [B, embedding_size]
            labels: Ground truth labels [B]

        Returns:
            loss: CosFace loss scalar
            logits: Cosine similarity logits [B, num_classes] (for accuracy computation)
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)  # [B, num_classes]

        # Apply margin to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # CosFace: subtract margin from target class cosine
        output = cosine - one_hot * self.margin

        # Scale
        output = output * self.scale

        # Cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss, cosine * self.scale


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss (alternative to CosFace).
    Included for experimentation.
    """

    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 50000,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos/sin of margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings_norm, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle edge case
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = F.cross_entropy(output, labels)

        return loss, cosine * self.scale


class FingerprintRecognitionModel(nn.Module):
    """
    Complete fingerprint recognition model combining backbone and loss.
    """

    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 50000,
        dropout: float = 0.0,
        remove_first_maxpool: bool = True,
        loss_type: str = "cosface",
        scale: float = 64.0,
        margin: float = 0.35,
        pretrained: bool = False,
    ):
        super().__init__()

        # Backbone
        self.backbone = ModifiedResNet18(
            embedding_size=embedding_size,
            dropout=dropout,
            remove_first_maxpool=remove_first_maxpool,
            pretrained=pretrained,
        )

        # Loss head
        if loss_type == "cosface":
            self.loss_head = CosFaceLoss(
                embedding_size=embedding_size,
                num_classes=num_classes,
                scale=scale,
                margin=margin,
            )
        elif loss_type == "arcface":
            self.loss_head = ArcFaceLoss(
                embedding_size=embedding_size,
                num_classes=num_classes,
                scale=scale,
                margin=margin,
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]
            labels: Ground truth labels [B] (optional, for training)

        Returns:
            embeddings: Feature embeddings [B, embedding_size]
            loss: Loss value (if labels provided)
            logits: Logits for accuracy computation (if labels provided)
        """
        embeddings = self.backbone(x)

        if labels is not None:
            loss, logits = self.loss_head(embeddings, labels)
            return embeddings, loss, logits
        else:
            return embeddings, None, None

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for inference."""
        return self.backbone(x)


def build_model(config) -> FingerprintRecognitionModel:
    """Build model from config."""
    return FingerprintRecognitionModel(
        embedding_size=config.model.embedding_size,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        remove_first_maxpool=config.model.remove_first_maxpool,
        loss_type="cosface",
        scale=config.cosface.scale,
        margin=config.cosface.margin,
        pretrained=False,
    )


if __name__ == "__main__":
    # Test model
    model = FingerprintRecognitionModel(
        embedding_size=512,
        num_classes=1000,
        remove_first_maxpool=True,
    )

    # Test forward pass
    x = torch.randn(4, 3, 299, 299)
    labels = torch.randint(0, 1000, (4,))

    embeddings, loss, logits = model(x, labels)

    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")

    # Test inference mode
    model.eval()
    with torch.no_grad():
        embeddings, _, _ = model(x)
        print(f"Inference embedding shape: {embeddings.shape}")
