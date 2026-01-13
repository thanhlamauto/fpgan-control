#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
Validation utilities for fingerprint recognition training.
Supports both synthetic (FPGAN) and real (FVC2004) validation sets.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add paths
_PWD = Path(__file__).absolute().parent
sys.path.append(str(_PWD))
sys.path.append(str(_PWD.parent / 'src'))

from config import Config


class SyntheticValidationDataset(Dataset):
    """
    Pre-generated synthetic validation dataset.

    Generates a fixed set of synthetic fingerprints at initialization
    for consistent validation during training.
    """

    def __init__(
        self,
        config: Config,
        num_identities: int = 100,
        images_per_identity: int = 8,
        seed: int = 12345,  # Different seed from training
    ):
        """
        Args:
            config: Configuration object
            num_identities: Number of synthetic identities for validation
            images_per_identity: Number of images per identity
            seed: Random seed for reproducible validation set
        """
        self.config = config
        self.num_identities = num_identities
        self.images_per_identity = images_per_identity
        self.seed = seed

        # Storage for pre-generated images
        self.images = []
        self.labels = []

        # Generate validation set
        self._generate_validation_set()

    def _generate_validation_set(self):
        """Generate fixed validation set from FPGAN."""
        print(f"Generating synthetic validation set: {self.num_identities} IDs x {self.images_per_identity} images...")

        # Import here to avoid circular imports
        from dataset import FPGANWrapper

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = FPGANWrapper(self.config.fpgan, device)
        generator.initialize()

        # Set seed for reproducibility
        torch.manual_seed(self.seed)

        # Pre-generate ID latents
        id_latents = torch.randn(
            self.num_identities,
            self.config.fpgan.id_latent_size,
            device=device
        )

        # Transform pipeline (same as training but no augmentation)
        transform = T.Compose([
            T.Resize((self.config.data.output_size, self.config.data.output_size), antialias=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        with torch.no_grad():
            for id_idx in tqdm(range(self.num_identities), desc="Generating synthetic val"):
                id_latent = id_latents[id_idx:id_idx+1]

                for img_idx in range(self.images_per_identity):
                    # Deterministic appearance latent for validation
                    torch.manual_seed(self.seed + id_idx * 1000 + img_idx)
                    app_latent = torch.randn(1, self.config.fpgan.app_latent_size, device=device)

                    # Generate image
                    image = generator.generate(id_latent, app_latent)
                    image = image.squeeze(0).cpu()

                    # Convert grayscale to RGB if needed
                    if image.shape[0] == 1:
                        image = image.repeat(3, 1, 1)

                    # Apply transform
                    image = transform(image)

                    self.images.append(image)
                    self.labels.append(id_idx)

        print(f"Generated {len(self.images)} synthetic validation images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx]


class FVC2004ValidationDataset(Dataset):
    """
    FVC2004 DB1_B validation dataset (real fingerprints).
    """

    def __init__(
        self,
        root_dir: str,
        output_size: int = 299,
    ):
        self.root_dir = Path(root_dir)
        self.output_size = output_size
        self.samples = []
        self.subjects = set()

        self._scan_directory()
        self.transform = self._build_transform()

    def _build_transform(self) -> nn.Module:
        return T.Compose([
            T.CenterCrop(480),  # Crop to square
            T.Resize((self.output_size, self.output_size), antialias=True),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _scan_directory(self):
        """Scan directory for FVC2004 format images."""
        import re
        pattern = re.compile(r'^(\d+)_(\d+)\.(tif|tiff|bmp|png|jpg)$', re.IGNORECASE)

        if not self.root_dir.exists():
            print(f"Warning: FVC2004 directory not found: {self.root_dir}")
            return

        for img_file in sorted(self.root_dir.iterdir()):
            if img_file.name.startswith('.'):
                continue
            match = pattern.match(img_file.name)
            if match:
                subject_id = int(match.group(1))
                impression_id = int(match.group(2))
                self.samples.append((str(img_file), subject_id, impression_id))
                self.subjects.add(subject_id)

        self.samples.sort(key=lambda x: (x[1], x[2]))
        self.subjects = sorted(self.subjects)
        self.subject_to_idx = {s: i for i, s in enumerate(self.subjects)}

        print(f"FVC2004 validation: {len(self.samples)} images, {len(self.subjects)} subjects")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, subject_id, _ = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.subject_to_idx[subject_id]
        return image, label


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """Compute Equal Error Rate."""
    from sklearn.metrics import roc_curve

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2

    return eer


def validate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str = "validation",
) -> Dict[str, float]:
    """
    Validate model on a dataset.

    Returns:
        Dictionary with EER and other metrics
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validating on {dataset_name}", leave=False):
            images = images.to(device)
            embeddings = model.get_embedding(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    # Compute similarity matrix
    sim_matrix = np.dot(embeddings, embeddings.T)

    # Get genuine and impostor scores
    n_samples = len(labels)
    genuine_scores = []
    impostor_scores = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            score = sim_matrix[i, j]
            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Compute EER
    eer = compute_eer(genuine_scores, impostor_scores)

    return {
        'eer': eer,
        'num_samples': n_samples,
        'num_subjects': len(np.unique(labels)),
        'num_genuine': len(genuine_scores),
        'num_impostor': len(impostor_scores),
    }


class ValidationManager:
    """
    Manages validation on multiple datasets during training.
    """

    def __init__(
        self,
        config: Config,
        device: torch.device,
        synthetic_val_ids: int = 100,
        synthetic_val_images_per_id: int = 8,
        fvc2004_db1b_path: Optional[str] = None,
    ):
        self.config = config
        self.device = device
        self.dataloaders = {}

        # Create synthetic validation set
        print("\n--- Setting up Synthetic Validation Set ---")
        try:
            synth_dataset = SyntheticValidationDataset(
                config=config,
                num_identities=synthetic_val_ids,
                images_per_identity=synthetic_val_images_per_id,
                seed=12345,
            )
            self.dataloaders['synthetic'] = DataLoader(
                synth_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        except Exception as e:
            print(f"Warning: Could not create synthetic validation set: {e}")

        # Create FVC2004 DB1_B validation set
        if fvc2004_db1b_path and os.path.exists(fvc2004_db1b_path):
            print("\n--- Setting up FVC2004 DB1_B Validation Set ---")
            fvc_dataset = FVC2004ValidationDataset(
                root_dir=fvc2004_db1b_path,
                output_size=config.data.output_size,
            )
            if len(fvc_dataset) > 0:
                self.dataloaders['fvc2004_db1b'] = DataLoader(
                    fvc_dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )
        else:
            print(f"FVC2004 DB1_B not found at {fvc2004_db1b_path}")

    def validate(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """
        Run validation on all configured datasets.

        Returns:
            Dictionary mapping dataset name to metrics
        """
        results = {}

        for name, dataloader in self.dataloaders.items():
            metrics = validate_model(model, dataloader, self.device, name)
            results[name] = metrics
            print(f"  {name}: EER={metrics['eer']:.4f} ({metrics['num_subjects']} subjects, {metrics['num_samples']} samples)")

        return results


if __name__ == "__main__":
    # Test validation
    from config import get_default_config

    config = get_default_config()
    config.data.num_identities = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test synthetic validation dataset
    val_manager = ValidationManager(
        config=config,
        device=device,
        synthetic_val_ids=50,
        synthetic_val_images_per_id=8,
        fvc2004_db1b_path="/kaggle/input/fvc-2004/FVC2004/Dbs/DB1_B",
    )

    print("\nValidation datasets ready:")
    for name in val_manager.dataloaders:
        print(f"  - {name}")
