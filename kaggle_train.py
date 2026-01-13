#!/usr/bin/env python
# coding: utf-8
"""
FPGAN-Control Synthetic Fingerprint Training Pipeline
======================================================

This script trains a fingerprint recognition model using synthetic data
generated on-the-fly by FPGAN-Control.

Evaluation: FVC2004 DB1_B (validation during training)
Test: FVC2004 DB1_A (final evaluation)

Usage on Kaggle:
1. Clone the repository
2. Download FPGAN model weights
3. Run this script

Environment: Kaggle with 2x T4 GPUs
"""

import os
import sys
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (adjust for Kaggle)
REPO_ROOT = Path("/kaggle/working/fpgan-control")
MODEL_URL = "https://1drv.ms/u/s!ArMf0NJWksTfiyJN7O70ezn7YAkJ?e=pl2WIA"
MODEL_DIR = REPO_ROOT / "models" / "id06fre20_fingers384_id_noise_same_id_idl005_posel000_large_pose_20230606-082209"

# FVC2004 Dataset paths on Kaggle
FVC2004_DB1_A = "/kaggle/input/fvc-2004/FVC2004/Dbs/DB1_A"  # Test set (100 subjects x 8 impressions)
FVC2004_DB1_B = "/kaggle/input/fvc-2004/FVC2004/Dbs/DB1_B"  # Validation set (10 subjects x 8 impressions)

# Training settings
NUM_IDENTITIES = 50000      # Number of synthetic identities
IMAGES_PER_ID = 11          # Images per identity
EPOCHS = 30                 # Training epochs
BATCH_SIZE = 128            # Batch size per GPU (2x T4 = 256 effective)
LEARNING_RATE = 0.1
SEED = 42

# Augmentation settings
ENABLE_PSEUDO_MIX = True    # Enable default + strong augmentation mix
MIX_RATIO = 0.5             # 50% default, 50% strong

# Output
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
RESULTS_DIR = "/kaggle/working/results"


# ============================================================================
# SETUP
# ============================================================================

def setup_environment():
    """Setup the environment and install dependencies."""
    print("=" * 60)
    print("SETTING UP ENVIRONMENT")
    print("=" * 60)

    # Install dependencies if needed
    try:
        import torchvision
        import sklearn
        import tqdm
    except ImportError:
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                       "torchvision", "scikit-learn", "tqdm"])

    # Check GPU
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("WARNING: No GPU available, training will be slow!")

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def download_model():
    """Download FPGAN model weights."""
    print("\n" + "=" * 60)
    print("DOWNLOADING MODEL WEIGHTS")
    print("=" * 60)

    if MODEL_DIR.exists() and (MODEL_DIR / "checkpoint").exists():
        print(f"Model already exists at {MODEL_DIR}")
        return True

    print(f"Please download the model manually from:\n{MODEL_URL}")
    print(f"And extract to: {MODEL_DIR}")
    print("\nAlternatively, upload the model as a Kaggle dataset.")

    # Check if model is available as Kaggle input
    kaggle_model_paths = [
        "/kaggle/input/fpgan-control-model",
        "/kaggle/input/fpgan-model",
        "/kaggle/input/fpgan-control-model/id06fre20_fingers384_id_noise_same_id_idl005_posel000_large_pose_20230606-082209",
    ]

    for path in kaggle_model_paths:
        if os.path.exists(path):
            print(f"Found model at {path}")
            # Create symlink
            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            if not MODEL_DIR.exists():
                # Check if path is the full model dir or parent
                if os.path.exists(os.path.join(path, "checkpoint")):
                    os.symlink(path, str(MODEL_DIR))
                else:
                    # Search for model dir inside
                    for item in os.listdir(path):
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "checkpoint")):
                            os.symlink(item_path, str(MODEL_DIR))
                            break
            return True

    return False


def check_fvc2004_dataset():
    """Check if FVC2004 dataset is available."""
    print("\n" + "=" * 60)
    print("CHECKING FVC2004 DATASET")
    print("=" * 60)

    db1_a_exists = os.path.exists(FVC2004_DB1_A)
    db1_b_exists = os.path.exists(FVC2004_DB1_B)

    print(f"FVC2004 DB1_A (Test): {'Found' if db1_a_exists else 'NOT FOUND'} at {FVC2004_DB1_A}")
    print(f"FVC2004 DB1_B (Val):  {'Found' if db1_b_exists else 'NOT FOUND'} at {FVC2004_DB1_B}")

    if db1_a_exists:
        num_files_a = len([f for f in os.listdir(FVC2004_DB1_A) if f.endswith('.tif')])
        print(f"  DB1_A: {num_files_a} .tif files")

    if db1_b_exists:
        num_files_b = len([f for f in os.listdir(FVC2004_DB1_B) if f.endswith('.tif')])
        print(f"  DB1_B: {num_files_b} .tif files")

    return db1_a_exists, db1_b_exists


def verify_setup():
    """Verify the setup is correct."""
    print("\n" + "=" * 60)
    print("VERIFYING SETUP")
    print("=" * 60)

    # Add paths
    sys.path.insert(0, str(REPO_ROOT / "training"))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    # Check model
    if not MODEL_DIR.exists():
        print("ERROR: Model directory not found!")
        print(f"Expected: {MODEL_DIR}")
        return False

    checkpoint_dir = MODEL_DIR / "checkpoint"
    if not checkpoint_dir.exists():
        print("ERROR: Checkpoint directory not found!")
        return False

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return False

    print(f"Found {len(checkpoints)} checkpoint(s)")

    # Verify generator
    print("\nVerifying FPGAN generator...")
    try:
        from config import get_default_config
        from dataset import verify_generator

        config = get_default_config()
        config.fpgan.model_dir = str(MODEL_DIR)

        verify_generator(config, num_samples=5)
        print("Generator verification PASSED!")
        return True

    except Exception as e:
        print(f"Generator verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TRAINING
# ============================================================================

def run_training(num_gpus: int):
    """Run the training."""
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    sys.path.insert(0, str(REPO_ROOT / "training"))

    from config import get_default_config
    from trainer import train_single_gpu, train_multi_gpu

    # Create config
    config = get_default_config()

    # FPGAN settings
    config.fpgan.model_dir = str(MODEL_DIR)

    # Data settings
    config.data.num_identities = NUM_IDENTITIES
    config.data.images_per_identity = IMAGES_PER_ID
    config.data.enable_pseudo_mix = ENABLE_PSEUDO_MIX
    config.data.mix_ratio = MIX_RATIO

    # Model settings
    config.model.num_classes = NUM_IDENTITIES
    config.model.embedding_size = 512
    config.model.remove_first_maxpool = True  # Per paper

    # Training settings
    config.training.epochs = EPOCHS
    config.training.batch_size = BATCH_SIZE
    config.training.lr = LEARNING_RATE
    config.training.seed = SEED
    config.training.checkpoint_dir = CHECKPOINT_DIR
    config.training.use_amp = True  # Mixed precision

    # Print config summary
    print("\nTraining Configuration:")
    print(f"  - Identities: {NUM_IDENTITIES:,}")
    print(f"  - Images per ID: {IMAGES_PER_ID}")
    print(f"  - Total samples: {NUM_IDENTITIES * IMAGES_PER_ID:,}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size per GPU: {BATCH_SIZE}")
    print(f"  - Effective batch size: {BATCH_SIZE * max(1, num_gpus)}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Pseudo-mix: {ENABLE_PSEUDO_MIX}")
    print(f"  - Mix ratio: {MIX_RATIO}")
    print(f"  - Seed: {SEED}")

    # Create output directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save config
    config.save(os.path.join(CHECKPOINT_DIR, "config.json"))

    # Start training
    if num_gpus > 1:
        print(f"\nStarting multi-GPU training on {num_gpus} GPUs...")
        train_multi_gpu(config, num_gpus)
    else:
        print("\nStarting single GPU training...")
        train_single_gpu(config)

    print("\nTraining completed!")
    return True


# ============================================================================
# EVALUATION
# ============================================================================

def run_validation():
    """Run validation on FVC2004 DB1_B during/after training."""
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION (FVC2004 DB1_B)")
    print("=" * 60)

    sys.path.insert(0, str(REPO_ROOT / "training"))

    from evaluate import evaluate, print_results_table

    # Check if validation set exists
    if not os.path.exists(FVC2004_DB1_B):
        print(f"WARNING: FVC2004 DB1_B not found at {FVC2004_DB1_B}")
        print("Skipping validation...")
        return None

    # Find best checkpoint
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "best.pth")
    if not os.path.exists(best_checkpoint):
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pth")
        if os.path.exists(latest_checkpoint):
            best_checkpoint = latest_checkpoint
        else:
            print("ERROR: No checkpoint found!")
            return None

    print(f"Using checkpoint: {best_checkpoint}")
    print(f"Validation set: {FVC2004_DB1_B}")

    # Run evaluation
    results = evaluate(
        checkpoint_path=best_checkpoint,
        test_path=FVC2004_DB1_B,
        batch_size=64,
        output_path=os.path.join(RESULTS_DIR, "validation_results_db1b.json"),
        dataset_type="fvc2004",
    )

    print_results_table(results)

    return results


def run_test():
    """Run final test on FVC2004 DB1_A."""
    print("\n" + "=" * 60)
    print("RUNNING TEST (FVC2004 DB1_A)")
    print("=" * 60)

    sys.path.insert(0, str(REPO_ROOT / "training"))

    from evaluate import evaluate, print_results_table

    # Check if test set exists
    if not os.path.exists(FVC2004_DB1_A):
        print(f"WARNING: FVC2004 DB1_A not found at {FVC2004_DB1_A}")
        print("Skipping test...")
        return None

    # Find best checkpoint
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "best.pth")
    if not os.path.exists(best_checkpoint):
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pth")
        if os.path.exists(latest_checkpoint):
            best_checkpoint = latest_checkpoint
        else:
            print("ERROR: No checkpoint found!")
            return None

    print(f"Using checkpoint: {best_checkpoint}")
    print(f"Test set: {FVC2004_DB1_A}")

    # Run evaluation
    results = evaluate(
        checkpoint_path=best_checkpoint,
        test_path=FVC2004_DB1_A,
        batch_size=64,
        output_path=os.path.join(RESULTS_DIR, "test_results_db1a.json"),
        dataset_type="fvc2004",
    )

    print_results_table(results)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("FPGAN-Control Synthetic Fingerprint Training")
    print("=" * 60)
    print("\nDataset Configuration:")
    print(f"  - Validation: FVC2004 DB1_B (10 subjects x 8 impressions)")
    print(f"  - Test: FVC2004 DB1_A (100 subjects x 8 impressions)")

    # Setup
    num_gpus = setup_environment()

    # Check FVC2004 dataset
    db1_a_exists, db1_b_exists = check_fvc2004_dataset()

    # Download model (or check if available)
    if not download_model():
        print("\nPlease download the model and try again.")
        print("Upload it as a Kaggle dataset named 'fpgan-control-model'")
        return

    # Verify setup
    if not verify_setup():
        print("\nSetup verification failed. Please check the errors above.")
        return

    # Train
    try:
        run_training(num_gpus)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validation on DB1_B
    val_results = None
    try:
        if db1_b_exists:
            val_results = run_validation()
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Test on DB1_A
    test_results = None
    try:
        if db1_a_exists:
            test_results = run_test()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    if val_results:
        print("\nValidation (FVC2004 DB1_B - 10 subjects):")
        print(f"  TAR @ FAR=0.1%:  {val_results.get('TAR@FAR=0.10%', 'N/A'):.4f}" if isinstance(val_results.get('TAR@FAR=0.10%'), float) else f"  TAR @ FAR=0.1%:  N/A")
        print(f"  TAR @ FAR=0.01%: {val_results.get('TAR@FAR=0.01%', 'N/A'):.4f}" if isinstance(val_results.get('TAR@FAR=0.01%'), float) else f"  TAR @ FAR=0.01%: N/A")
        print(f"  EER: {val_results.get('EER', 'N/A'):.4f}" if isinstance(val_results.get('EER'), float) else f"  EER: N/A")

    if test_results:
        print("\nTest (FVC2004 DB1_A - 100 subjects):")
        print(f"  TAR @ FAR=0.1%:  {test_results.get('TAR@FAR=0.10%', 'N/A'):.4f}" if isinstance(test_results.get('TAR@FAR=0.10%'), float) else f"  TAR @ FAR=0.1%:  N/A")
        print(f"  TAR @ FAR=0.01%: {test_results.get('TAR@FAR=0.01%', 'N/A'):.4f}" if isinstance(test_results.get('TAR@FAR=0.01%'), float) else f"  TAR @ FAR=0.01%: N/A")
        print(f"  EER: {test_results.get('EER', 'N/A'):.4f}" if isinstance(test_results.get('EER'), float) else f"  EER: N/A")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
