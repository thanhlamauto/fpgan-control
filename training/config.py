#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
Configuration for FPGAN-Control synthetic fingerprint training pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import os


@dataclass
class FPGANConfig:
    """FPGAN Generator configuration."""
    model_dir: str = "models/id06fre20_fingers384_id_noise_same_id_idl005_posel000_large_pose_20230606-082209"
    latent_size: int = 512
    id_latent_size: int = 256
    app_latent_size: int = 256
    image_size: int = 384
    truncation_default: float = 1.0  # No truncation
    truncation_strong: float = 0.7   # For strong branch


@dataclass
class DataConfig:
    """Dataset configuration."""
    num_identities: int = 50000
    images_per_identity: int = 11
    output_size: int = 299  # ResNet input size
    mix_ratio: float = 0.5  # 50% default, 50% strong
    enable_pseudo_mix: bool = True

    # Augmentation - Default branch
    aug_default_rotation: float = 15.0
    aug_default_translate: float = 0.08
    aug_default_scale: tuple = (0.95, 1.05)

    # Augmentation - Strong branch
    aug_strong_rotation: float = 30.0
    aug_strong_translate: float = 0.15
    aug_strong_scale: tuple = (0.85, 1.15)
    aug_strong_blur_prob: float = 0.3
    aug_strong_noise_std: float = 0.02


@dataclass
class ModelConfig:
    """Recognition model configuration."""
    backbone: str = "resnet18"
    embedding_size: int = 512
    num_classes: int = 50000  # Same as num_identities
    dropout: float = 0.0
    remove_first_maxpool: bool = True  # Per paper


@dataclass
class CosFaceConfig:
    """CosFace loss configuration."""
    scale: float = 64.0
    margin: float = 0.35


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 64  # Per GPU (reduced for on-the-fly generation)
    num_workers: int = 4
    epochs: int = 30
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: List[int] = field(default_factory=lambda: [10, 20, 25])
    lr_gamma: float = 0.1

    # Reproducibility
    seed: int = 42

    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_every: int = 100


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # FVC2004 paths on Kaggle
    fvc2004_db1_a: str = "/kaggle/input/fvc-2004/FVC2004/Dbs/DB1_A"  # Test set (100 subjects)
    fvc2004_db1_b: str = "/kaggle/input/fvc-2004/FVC2004/Dbs/DB1_B"  # Validation set (10 subjects)
    far_thresholds: List[float] = field(default_factory=lambda: [0.001, 0.0001])  # FAR=0.1%, 0.01%
    batch_size: int = 64


@dataclass
class Config:
    """Main configuration."""
    fpgan: FPGANConfig = field(default_factory=FPGANConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cosface: CosFaceConfig = field(default_factory=CosFaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def save(self, path: str):
        """Save config to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=lambda x: x.__dict__)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        config.fpgan = FPGANConfig(**data.get('fpgan', {}))
        config.data = DataConfig(**data.get('data', {}))
        config.model = ModelConfig(**data.get('model', {}))
        config.cosface = CosFaceConfig(**data.get('cosface', {}))
        config.training = TrainingConfig(**data.get('training', {}))
        config.eval = EvalConfig(**data.get('eval', {}))
        return config


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
