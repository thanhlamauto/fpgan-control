#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
Evaluation script for fingerprint verification on FVC2004 dataset.
Reports TAR @ FAR=0.1% and FAR=0.01%.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve

# Add paths
_PWD = Path(__file__).absolute().parent
sys.path.append(str(_PWD))

from config import Config, get_default_config
from models import FingerprintRecognitionModel, build_model


class FVC2004Dataset(Dataset):
    """
    FVC2004 Dataset loader.

    File naming convention: {subject_id}_{impression_id}.tif
    Example: 1_1.tif, 1_2.tif, ..., 100_8.tif

    DB1_A: 100 subjects × 8 impressions = 800 images (test)
    DB1_B: 10 subjects × 8 impressions = 80 images (validation)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[nn.Module] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()

        # Scan directory
        self.samples = []  # List of (image_path, subject_id, impression_id)
        self.subjects = set()

        self._scan_directory()

    def _default_transform(self) -> nn.Module:
        return T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Grayscale to RGB
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _scan_directory(self):
        """Scan directory for FVC2004 format images."""
        if not self.root_dir.exists():
            raise ValueError(f"FVC2004 directory not found: {self.root_dir}")

        # Pattern: {subject}_{impression}.tif
        pattern = re.compile(r'^(\d+)_(\d+)\.(tif|tiff|bmp|png|jpg)$', re.IGNORECASE)

        for img_file in sorted(self.root_dir.iterdir()):
            if img_file.name.startswith('.'):  # Skip hidden files
                continue

            match = pattern.match(img_file.name)
            if match:
                subject_id = int(match.group(1))
                impression_id = int(match.group(2))
                self.samples.append((str(img_file), subject_id, impression_id))
                self.subjects.add(subject_id)

        # Sort by subject then impression
        self.samples.sort(key=lambda x: (x[1], x[2]))
        self.subjects = sorted(self.subjects)

        # Create subject to index mapping (0-indexed)
        self.subject_to_idx = {s: i for i, s in enumerate(self.subjects)}

        print(f"Found {len(self.samples)} images from {len(self.subjects)} subjects")
        print(f"Subjects: {min(self.subjects)} to {max(self.subjects)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, subject_id, impression_id = self.samples[idx]

        # Load image
        image = Image.open(img_path)

        # Convert to RGB if grayscale
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transform(image)

        # Return 0-indexed subject label
        label = self.subject_to_idx[subject_id]

        return image, label, img_path


class N2NTestDataset(Dataset):
    """
    NIST SD302 (N2N) Test Dataset loader (kept for compatibility).
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[nn.Module] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()
        self.samples = []
        self.subjects = []
        self.subject_to_id = {}
        self._scan_directory()

    def _default_transform(self) -> nn.Module:
        return T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _scan_directory(self):
        if not self.root_dir.exists():
            raise ValueError(f"N2N test directory not found: {self.root_dir}")

        subject_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        for subject_idx, subject_dir in enumerate(subject_dirs):
            subject_name = subject_dir.name
            self.subjects.append(subject_name)
            self.subject_to_id[subject_name] = subject_idx

            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
                for img_file in sorted(subject_dir.glob(ext)):
                    self.samples.append((str(img_file), subject_idx))

        print(f"Found {len(self.samples)} images from {len(self.subjects)} subjects")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, subject_id = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, subject_id, img_path


def get_dataset(root_dir: str, dataset_type: str = "auto") -> Dataset:
    """
    Get appropriate dataset based on type or auto-detect.

    Args:
        root_dir: Path to dataset
        dataset_type: "fvc2004", "n2n", or "auto"
    """
    root_path = Path(root_dir)

    if dataset_type == "auto":
        # Auto-detect based on structure
        # FVC2004: files directly in folder with pattern {id}_{imp}.tif
        # N2N: subdirectories per subject
        has_subdirs = any(p.is_dir() for p in root_path.iterdir() if not p.name.startswith('.'))
        has_tif_files = any(p.suffix.lower() in ['.tif', '.tiff'] for p in root_path.iterdir())

        if has_tif_files and not has_subdirs:
            dataset_type = "fvc2004"
        else:
            dataset_type = "n2n"

        print(f"Auto-detected dataset type: {dataset_type}")

    if dataset_type == "fvc2004":
        return FVC2004Dataset(root_dir)
    else:
        return N2NTestDataset(root_dir)


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract embeddings from all images.

    Returns:
        embeddings: [N, embedding_size]
        labels: [N]
        paths: List of image paths
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)

            # Get embeddings
            embeddings = model.get_embedding(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            all_paths.extend(paths)

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    return embeddings, labels, all_paths


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    return np.dot(embeddings, embeddings.T)


def compute_verification_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    far_targets: List[float] = [0.001, 0.0001],
) -> Dict[str, float]:
    """
    Compute verification metrics (TAR @ FAR).

    Args:
        embeddings: [N, embedding_size] normalized embeddings
        labels: [N] subject labels
        far_targets: Target FAR values

    Returns:
        Dictionary with TAR values for each FAR target
    """
    n_samples = len(labels)

    # Compute similarity matrix
    print("Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(embeddings)

    # Get genuine and impostor pairs
    print("Generating pairs...")
    genuine_scores = []
    impostor_scores = []

    # Use upper triangle only (avoid duplicate pairs)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            score = sim_matrix[i, j]
            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print(f"Genuine pairs: {len(genuine_scores)}")
    print(f"Impostor pairs: {len(impostor_scores)}")

    # Create labels for ROC curve
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find TAR at target FARs
    results = {}
    for far_target in far_targets:
        # Find threshold for target FAR
        idx = np.argmin(np.abs(fpr - far_target))
        tar = tpr[idx]
        actual_far = fpr[idx]
        threshold = thresholds[idx]

        far_str = f"{far_target * 100:.2f}%"
        results[f"TAR@FAR={far_str}"] = tar
        results[f"ActualFAR@FAR={far_str}"] = actual_far
        results[f"Threshold@FAR={far_str}"] = threshold

        print(f"TAR @ FAR={far_str}: {tar:.4f} (actual FAR: {actual_far:.6f}, threshold: {threshold:.4f})")

    # Also compute EER
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
    results["EER"] = eer
    print(f"EER: {eer:.4f}")

    return results


def evaluate(
    checkpoint_path: str,
    test_path: str,
    config: Optional[Config] = None,
    batch_size: int = 64,
    output_path: Optional[str] = None,
    dataset_type: str = "auto",
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        checkpoint_path: Path to model checkpoint
        test_path: Path to test set (FVC2004 or N2N)
        config: Optional config (loaded from checkpoint if not provided)
        batch_size: Batch size for embedding extraction
        output_path: Optional path to save results
        dataset_type: "fvc2004", "n2n", or "auto"

    Returns:
        Dictionary with evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build model
    if config is None:
        config = get_default_config()

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training accuracy: {checkpoint.get('best_acc', 'unknown')}")

    # Create dataset
    print(f"Loading test set from {test_path}...")
    test_dataset = get_dataset(test_path, dataset_type)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Extract embeddings
    embeddings, labels, paths = extract_embeddings(model, test_loader, device)

    # Compute metrics
    print("\nComputing verification metrics...")
    results = compute_verification_metrics(
        embeddings,
        labels,
        far_targets=[0.001, 0.0001]  # FAR=0.1%, 0.01%
    )

    # Add metadata
    results['checkpoint'] = checkpoint_path
    results['test_path'] = test_path
    results['dataset_type'] = dataset_type
    results['num_samples'] = len(labels)
    results['num_subjects'] = len(np.unique(labels))
    results['training_epoch'] = checkpoint.get('epoch', 'unknown')
    results['training_accuracy'] = checkpoint.get('best_acc', 'unknown')

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return results


def print_results_table(results: Dict[str, float]):
    """Print results in a nice table format."""
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':<20}")
    print("-" * 60)

    # Key metrics first
    key_metrics = [
        "TAR@FAR=0.10%",
        "TAR@FAR=0.01%",
        "EER",
    ]

    for metric in key_metrics:
        if metric in results:
            value = results[metric]
            if isinstance(value, float):
                print(f"{metric:<30} {value:.4f}")
            else:
                print(f"{metric:<30} {value}")

    print("-" * 60)

    # Dataset info
    print(f"{'Dataset':<30} {results.get('test_path', 'N/A')}")
    print(f"{'Samples':<30} {results.get('num_samples', 'N/A')}")
    print(f"{'Subjects':<30} {results.get('num_subjects', 'N/A')}")
    print(f"{'Training Epoch':<30} {results.get('training_epoch', 'N/A')}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate fingerprint verification model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-path', type=str, required=True, help='Path to test set')
    parser.add_argument('--dataset-type', type=str, default='auto',
                       choices=['auto', 'fvc2004', 'n2n'], help='Dataset type')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json', help='Output path')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        config = Config.load(args.config)

    # Evaluate
    results = evaluate(
        checkpoint_path=args.checkpoint,
        test_path=args.test_path,
        config=config,
        batch_size=args.batch_size,
        output_path=args.output,
        dataset_type=args.dataset_type,
    )

    # Print table
    print_results_table(results)


if __name__ == "__main__":
    main()
