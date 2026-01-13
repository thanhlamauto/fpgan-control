#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
On-the-fly synthetic fingerprint dataset using FPGAN-Control generator.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random
from typing import Optional, Tuple, Dict, List
import sys
from pathlib import Path

# Add FPGAN source to path
_PWD = Path(__file__).absolute().parent.parent
sys.path.append(str(_PWD / 'src'))

from config import Config, DataConfig, FPGANConfig


class FPGANWrapper:
    """
    Wrapper for FPGAN generator to handle on-the-fly generation.
    Designed for multi-GPU usage with careful device management.
    """

    def __init__(self, config: FPGANConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.batch_utils = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization - call this after forking in DataLoader workers."""
        if self._initialized:
            return

        from evaluation.inference_class import Inference

        inference = Inference(self.config.model_dir)
        self.model = inference.model
        self.batch_utils = inference.batch_utils
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True

    @torch.no_grad()
    def generate(
        self,
        id_latent: torch.Tensor,
        app_latent: torch.Tensor,
        noise: Optional[List[torch.Tensor]] = None,
        truncation: float = 1.0
    ) -> torch.Tensor:
        """
        Generate fingerprint images.

        Args:
            id_latent: Identity latent [B, 256]
            app_latent: Appearance latent [B, 256]
            noise: Optional noise tensors
            truncation: Truncation value (1.0 = no truncation)

        Returns:
            Generated images [B, C, H, W] in range [0, 1]
        """
        if not self._initialized:
            self.initialize()

        batch_size = id_latent.shape[0]

        # Concatenate latents
        latent = torch.cat([id_latent, app_latent], dim=1).to(self.device)

        # Generate noise if not provided
        if noise is None:
            if hasattr(self.model, 'module'):
                noise = self.model.module.make_noise(device=self.device)
            else:
                noise = self.model.make_noise(device=self.device)
            # Expand noise for batch
            noise = [n.expand(batch_size, -1, -1, -1).clone() for n in noise]

        # Generate images
        images, _ = self.model([latent], noise=noise)

        # Normalize to [0, 1]
        images = images.mul(0.5).add(0.5).clamp(min=0., max=1.)

        return images


class SyntheticFingerprintDataset(Dataset):
    """
    On-the-fly synthetic fingerprint dataset.

    Generates fingerprints using FPGAN-Control during training.
    Each identity has a fixed ID latent, with random appearance latents per sample.
    """

    def __init__(
        self,
        config: Config,
        generator: FPGANWrapper,
        branch: str = "default",  # "default" or "strong"
        transform: Optional[nn.Module] = None,
    ):
        """
        Args:
            config: Configuration object
            generator: FPGAN generator wrapper
            branch: "default" or "strong" for augmentation branch
            transform: Additional transforms to apply
        """
        self.config = config
        self.data_config = config.data
        self.fpgan_config = config.fpgan
        self.generator = generator
        self.branch = branch
        self.transform = transform

        # Total samples = num_identities * images_per_identity
        self.num_samples = self.data_config.num_identities * self.data_config.images_per_identity

        # Pre-generate ID latents for all identities (fixed for consistency)
        # Use deterministic seed for reproducibility
        rng = torch.Generator()
        rng.manual_seed(config.training.seed)
        self.id_latents = torch.randn(
            self.data_config.num_identities,
            self.fpgan_config.id_latent_size,
            generator=rng
        )

        # Pre-generate ID noise for each identity (same noise for same ID)
        self.id_noise_seeds = torch.randint(
            0, 2**31,
            (self.data_config.num_identities,),
            generator=rng
        )

        # Build augmentation pipeline
        self.augmentation = self._build_augmentation()

        # Truncation based on branch
        self.truncation = (
            self.fpgan_config.truncation_default
            if branch == "default"
            else self.fpgan_config.truncation_strong
        )

    def _build_augmentation(self) -> nn.Module:
        """Build augmentation pipeline based on branch type."""
        cfg = self.data_config

        if self.branch == "default":
            rotation = cfg.aug_default_rotation
            translate = cfg.aug_default_translate
            scale = cfg.aug_default_scale
            blur_prob = 0.0
            noise_std = 0.0
        else:  # strong
            rotation = cfg.aug_strong_rotation
            translate = cfg.aug_strong_translate
            scale = cfg.aug_strong_scale
            blur_prob = cfg.aug_strong_blur_prob
            noise_std = cfg.aug_strong_noise_std

        transforms_list = [
            # Resize to model input size
            T.Resize((cfg.output_size, cfg.output_size), antialias=True),
            # Random affine
            T.RandomAffine(
                degrees=rotation,
                translate=(translate, translate),
                scale=scale,
                fill=1.0  # White fill for fingerprints
            ),
        ]

        return T.Compose(transforms_list), blur_prob, noise_std

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a single sample.

        Returns:
            image: Fingerprint image [C, H, W]
            label: Identity label (class index)
            metadata: Dict with additional info
        """
        # Calculate identity and impression index
        identity_idx = idx // self.data_config.images_per_identity
        impression_idx = idx % self.data_config.images_per_identity

        # Get fixed ID latent for this identity
        id_latent = self.id_latents[identity_idx].unsqueeze(0)

        # Generate random appearance latent (different each time)
        app_latent = torch.randn(1, self.fpgan_config.app_latent_size)

        # Generate noise deterministically for this identity
        noise_rng = torch.Generator()
        noise_rng.manual_seed(int(self.id_noise_seeds[identity_idx].item()))

        # Generate image
        with torch.no_grad():
            image = self.generator.generate(
                id_latent=id_latent,
                app_latent=app_latent,
                truncation=self.truncation
            )
            image = image.squeeze(0).cpu()

        # Apply augmentation
        aug_transforms, blur_prob, noise_std = self.augmentation
        image = aug_transforms(image)

        # Strong branch: additional blur and noise
        if self.branch == "strong":
            if random.random() < blur_prob:
                kernel_size = random.choice([3, 5])
                image = TF.gaussian_blur(image, kernel_size=kernel_size)

            if noise_std > 0:
                noise = torch.randn_like(image) * noise_std
                image = (image + noise).clamp(0, 1)

        # Apply additional transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Normalize for model input
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

        metadata = {
            "identity_id": identity_idx,
            "impression_id": impression_idx,
            "branch": self.branch,
            "tag": f"FPGC-0_{self.branch}"
        }

        return image, identity_idx, metadata


class MixedSyntheticDataset(Dataset):
    """
    Mixed dataset combining default and strong augmentation branches.
    """

    def __init__(
        self,
        config: Config,
        generator: FPGANWrapper,
    ):
        self.config = config
        self.mix_ratio = config.data.mix_ratio  # Ratio of default branch

        # Create both branches
        self.default_dataset = SyntheticFingerprintDataset(
            config, generator, branch="default"
        )
        self.strong_dataset = SyntheticFingerprintDataset(
            config, generator, branch="strong"
        )

        self.total_samples = len(self.default_dataset)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        # Randomly choose branch based on mix_ratio
        if random.random() < self.mix_ratio:
            return self.default_dataset[idx]
        else:
            return self.strong_dataset[idx]


class OnTheFlyDataLoader:
    """
    DataLoader wrapper that handles FPGAN generation on multiple GPUs.
    """

    def __init__(
        self,
        config: Config,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # Determine device for this process
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        # Create generator wrapper
        self.generator = FPGANWrapper(config.fpgan, self.device)

    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        if self.config.data.enable_pseudo_mix:
            dataset = MixedSyntheticDataset(self.config, self.generator)
        else:
            dataset = SyntheticFingerprintDataset(
                self.config, self.generator, branch="default"
            )

        # Create sampler for distributed training
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.config.training.seed
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,  # Must be 0 for on-the-fly generation with CUDA
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(batch):
        """Custom collate function to handle metadata."""
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        metadata = [item[2] for item in batch]
        return images, labels, metadata


def verify_generator(config: Config, num_samples: int = 5):
    """
    Verify FPGAN generator by generating sample images.

    Args:
        config: Configuration object
        num_samples: Number of samples to generate
    """
    import torchvision.utils as vutils
    import os

    print(f"Verifying FPGAN generator with {num_samples} samples...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = FPGANWrapper(config.fpgan, device)
    generator.initialize()

    # Generate samples
    id_latent = torch.randn(1, config.fpgan.id_latent_size, device=device)
    images = []

    for i in range(num_samples):
        app_latent = torch.randn(1, config.fpgan.app_latent_size, device=device)
        img = generator.generate(id_latent, app_latent)
        images.append(img)

    # Save grid
    grid = vutils.make_grid(torch.cat(images, dim=0), nrow=num_samples, padding=2)

    os.makedirs("verify_output", exist_ok=True)
    vutils.save_image(grid, "verify_output/fpgan_samples.png")
    print(f"Saved verification samples to verify_output/fpgan_samples.png")

    return True


if __name__ == "__main__":
    # Test the dataset
    from config import get_default_config

    config = get_default_config()
    config.data.num_identities = 100  # Small test

    verify_generator(config, num_samples=5)
