#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
Training script for fingerprint recognition with on-the-fly FPGAN generation.
Supports multi-GPU training on Kaggle (2x T4).
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Add paths
_PWD = Path(__file__).absolute().parent
sys.path.append(str(_PWD))
sys.path.append(str(_PWD.parent / 'src'))

from config import Config, get_default_config
from models import FingerprintRecognitionModel, build_model
from dataset import OnTheFlyDataLoader, verify_generator


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """
    Trainer class for fingerprint recognition model.
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
        self.is_main = (rank == 0)

        # Device
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        # Set seed
        set_seed(config.training.seed + rank)

        # Build model
        self.model = build_model(config).to(self.device)

        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])

        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.training.lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.training.lr_milestones,
            gamma=config.training.lr_gamma,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None

        # Data loader
        self.data_loader_wrapper = OnTheFlyDataLoader(config, rank, world_size)
        self.train_loader = None  # Created lazily

        # Logging
        self.log_dir = Path(config.training.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_log = []
        self.best_acc = 0.0
        self.start_epoch = 0

    def log(self, msg: str):
        """Log message (only on main process)."""
        if self.is_main:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_main:
            return

        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_acc': self.best_acc,
            'config': self.config.__dict__,
            'training_log': self.training_log,
        }

        # Save latest
        torch.save(checkpoint, self.log_dir / 'latest.pth')

        # Save periodic
        if epoch % self.config.training.save_every == 0:
            torch.save(checkpoint, self.log_dir / f'epoch_{epoch:03d}.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, self.log_dir / 'best.pth')
            self.log(f"Saved best model with accuracy: {self.best_acc:.4f}")

    def load_checkpoint(self, path: Optional[str] = None):
        """Load model checkpoint."""
        if path is None:
            path = self.log_dir / 'latest.pth'

        if not Path(path).exists():
            self.log(f"No checkpoint found at {path}")
            return

        self.log(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        model = self.model.module if self.world_size > 1 else self.model
        model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.start_epoch = checkpoint['epoch']
        self.training_log = checkpoint.get('training_log', [])

        self.log(f"Resumed from epoch {self.start_epoch}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        batch_time = AverageMeter()

        # Create data loader if not exists
        if self.train_loader is None:
            self.train_loader = self.data_loader_wrapper.get_train_loader()

        # Set epoch for distributed sampler
        if self.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        num_batches = len(self.train_loader)

        for batch_idx, (images, labels, metadata) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.training.use_amp):
                embeddings, loss, logits = self.model(images, labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean()

            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Log
            if self.is_main and (batch_idx + 1) % self.config.training.log_every == 0:
                self.log(
                    f"Epoch [{epoch}/{self.config.training.epochs}] "
                    f"Batch [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"Acc: {acc_meter.avg:.4f} "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f} "
                    f"Time: {batch_time.avg:.3f}s/batch"
                )

        return {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg,
        }

    def train(self, resume: bool = True):
        """Full training loop."""
        # Save config
        if self.is_main:
            self.config.save(str(self.log_dir / 'config.json'))

        # Optionally resume
        if resume:
            self.load_checkpoint()

        self.log(f"Starting training from epoch {self.start_epoch + 1}")
        self.log(f"Total epochs: {self.config.training.epochs}")
        self.log(f"Batch size per GPU: {self.config.training.batch_size}")
        self.log(f"World size: {self.world_size}")
        self.log(f"Effective batch size: {self.config.training.batch_size * self.world_size}")

        for epoch in range(self.start_epoch + 1, self.config.training.epochs + 1):
            epoch_start = time.time()

            # Train
            metrics = self.train_epoch(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log epoch results
            epoch_time = time.time() - epoch_start
            self.log(
                f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}"
            )

            # Save training log
            log_entry = {
                'epoch': epoch,
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'lr': self.scheduler.get_last_lr()[0],
                'time': epoch_time,
            }
            self.training_log.append(log_entry)

            # Check if best
            is_best = metrics['accuracy'] > self.best_acc
            if is_best:
                self.best_acc = metrics['accuracy']

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Sync between processes
            if self.world_size > 1:
                dist.barrier()

        self.log(f"Training completed! Best accuracy: {self.best_acc:.4f}")

        # Save final training log
        if self.is_main:
            with open(self.log_dir / 'training_log.json', 'w') as f:
                json.dump(self.training_log, f, indent=2)


def train_single_gpu(config: Config):
    """Single GPU training."""
    trainer = Trainer(config, rank=0, world_size=1)
    trainer.train()


def train_worker(rank: int, world_size: int, config: Config):
    """Worker function for distributed training."""
    setup_distributed(rank, world_size)

    trainer = Trainer(config, rank=rank, world_size=world_size)
    trainer.train()

    cleanup_distributed()


def train_multi_gpu(config: Config, num_gpus: int = 2):
    """Multi-GPU training using spawn."""
    import torch.multiprocessing as mp

    mp.spawn(
        train_worker,
        args=(num_gpus, config),
        nprocs=num_gpus,
        join=True
    )


def main():
    parser = argparse.ArgumentParser(description='Train fingerprint recognition model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--num-ids', type=int, default=50000, help='Number of identities')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--model-dir', type=str, default=None, help='FPGAN model directory')
    parser.add_argument('--no-mix', action='store_true', help='Disable pseudo-mix augmentation')
    parser.add_argument('--mix-ratio', type=float, default=0.5, help='Mix ratio (default branch ratio)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verify', action='store_true', help='Verify generator before training')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs (auto-detect if not set)')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = get_default_config()

    # Override config with arguments
    config.data.num_identities = args.num_ids
    config.model.num_classes = args.num_ids
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr
    config.training.checkpoint_dir = args.checkpoint_dir
    config.training.seed = args.seed
    config.data.enable_pseudo_mix = not args.no_mix
    config.data.mix_ratio = args.mix_ratio

    if args.model_dir:
        config.fpgan.model_dir = args.model_dir

    # Detect GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    print(f"Detected {num_gpus} GPU(s)")

    # Verify generator
    if args.verify:
        print("Verifying FPGAN generator...")
        verify_generator(config, num_samples=5)
        print("Generator verification complete!")

    # Train
    if num_gpus > 1:
        print(f"Starting multi-GPU training on {num_gpus} GPUs...")
        train_multi_gpu(config, num_gpus)
    else:
        print("Starting single GPU training...")
        train_single_gpu(config)


if __name__ == "__main__":
    main()
