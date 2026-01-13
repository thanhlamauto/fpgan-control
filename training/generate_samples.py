#!/usr/bin/env python
#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
Generate and save sample synthetic fingerprints for visualization.
Also exports metadata CSV for reproducibility.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

# Add paths
_PWD = Path(__file__).absolute().parent
sys.path.append(str(_PWD))
sys.path.append(str(_PWD.parent / 'src'))

from config import Config, get_default_config
from dataset import FPGANWrapper


def generate_samples(
    config: Config,
    output_dir: str,
    num_identities: int = 100,
    images_per_identity: int = 11,
    seed: int = 42,
    save_grid: bool = True,
):
    """
    Generate and save synthetic fingerprint samples.

    Args:
        config: Configuration object
        output_dir: Output directory
        num_identities: Number of identities to generate
        images_per_identity: Number of images per identity
        seed: Random seed for reproducibility
        save_grid: Whether to save a visualization grid
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    print("Initializing FPGAN generator...")
    generator = FPGANWrapper(config.fpgan, device)
    generator.initialize()

    # Pre-generate ID latents
    print(f"Pre-generating {num_identities} identity latents...")
    id_latents = torch.randn(num_identities, config.fpgan.id_latent_size, device=device)

    # Metadata for CSV
    metadata_rows = []

    # Generate images
    print(f"\nGenerating {num_identities} identities x {images_per_identity} images...")

    # For visualization grid
    grid_images = []

    for id_idx in tqdm(range(num_identities), desc="Generating identities"):
        # Create identity directory
        id_dir = output_path / f"identity_{id_idx:06d}"
        id_dir.mkdir(exist_ok=True)

        id_latent = id_latents[id_idx:id_idx+1]

        for img_idx in range(images_per_identity):
            # Generate random appearance latent
            app_seed = seed + id_idx * 1000 + img_idx
            torch.manual_seed(app_seed)
            app_latent = torch.randn(1, config.fpgan.app_latent_size, device=device)

            # Generate image
            with torch.no_grad():
                image = generator.generate(id_latent, app_latent)
                image = image.squeeze(0).cpu()

            # Save image
            img_path = id_dir / f"img_{img_idx:03d}.png"
            save_image(image, str(img_path))

            # Add to metadata
            metadata_rows.append({
                'identity_id': id_idx,
                'impression_id': img_idx,
                'filepath': str(img_path.relative_to(output_path)),
                'tag': 'FPGC-0',
                'id_seed': seed,
                'app_seed': app_seed,
            })

            # Add first few to grid
            if id_idx < 10 and img_idx < 5:
                grid_images.append(image)

    # Save metadata CSV
    csv_path = output_path / "metadata.csv"
    print(f"\nSaving metadata to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['identity_id', 'impression_id', 'filepath', 'tag', 'id_seed', 'app_seed'])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Save metadata JSON
    json_path = output_path / "metadata.json"
    metadata_json = {
        'num_identities': num_identities,
        'images_per_identity': images_per_identity,
        'total_images': num_identities * images_per_identity,
        'seed': seed,
        'model_dir': config.fpgan.model_dir,
        'generated_at': datetime.now().isoformat(),
        'samples': metadata_rows,
    }
    with open(json_path, 'w') as f:
        json.dump(metadata_json, f, indent=2)

    # Save visualization grid
    if save_grid and grid_images:
        print("Saving visualization grid...")
        grid = torch.stack(grid_images)
        # Reshape to (10 identities, 5 impressions, C, H, W) -> grid
        from torchvision.utils import make_grid
        grid_img = make_grid(grid, nrow=5, padding=2, normalize=False)
        save_image(grid_img, str(output_path / "visualization_grid.png"))

    print(f"\nGeneration complete!")
    print(f"  Total identities: {num_identities}")
    print(f"  Total images: {num_identities * images_per_identity}")
    print(f"  Output directory: {output_path}")
    print(f"  Metadata CSV: {csv_path}")
    print(f"  Metadata JSON: {json_path}")

    return metadata_json


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic fingerprint samples')
    parser.add_argument('--model-dir', type=str, required=True, help='Path to FPGAN model directory')
    parser.add_argument('--output-dir', type=str, default='synthetic_samples', help='Output directory')
    parser.add_argument('--num-ids', type=int, default=100, help='Number of identities')
    parser.add_argument('--images-per-id', type=int, default=11, help='Images per identity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-grid', action='store_true', help='Skip visualization grid')

    args = parser.parse_args()

    # Create config
    config = get_default_config()
    config.fpgan.model_dir = args.model_dir

    # Generate samples
    generate_samples(
        config=config,
        output_dir=args.output_dir,
        num_identities=args.num_ids,
        images_per_identity=args.images_per_id,
        seed=args.seed,
        save_grid=not args.no_grid,
    )


if __name__ == "__main__":
    main()
