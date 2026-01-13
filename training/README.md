# FPGAN-Control Training Pipeline

Train fingerprint recognition models using synthetic data generated on-the-fly by FPGAN-Control.

## Features

- **On-the-fly generation**: No need to save large datasets - fingerprints are generated during training
- **Modified ResNet-18**: Removed first max-pooling layer as per paper recommendations
- **CosFace loss**: Large margin cosine loss for robust embedding learning
- **Multi-GPU support**: Distributed training on 2x T4 GPUs (Kaggle)
- **Pseudo-mix augmentation**: Combine default and strong augmentation branches
- **Reproducible**: Seeded random generation, config logging, checkpoint saving

## Quick Start on Kaggle

### 1. Setup Kaggle Notebook

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/fpgan-control.git
%cd fpgan-control
```

### 2. Download FPGAN Model

Upload the pre-trained FPGAN model as a Kaggle dataset, or download directly:

```python
# Option 1: If you uploaded as Kaggle dataset
!ln -s /kaggle/input/fpgan-model models/id06fre20_fingers384_id_noise_same_id_idl005_posel000_large_pose_20230606-082209

# Option 2: Download from OneDrive (manual step required)
```

### 3. Run Training

```python
# Run the main training script
!python kaggle_train.py
```

Or use individual components:

```python
import sys
sys.path.insert(0, 'training')
sys.path.insert(0, 'src')

from config import get_default_config
from trainer import train_multi_gpu

config = get_default_config()
config.data.num_identities = 50000
config.training.epochs = 30

train_multi_gpu(config, num_gpus=2)
```

## File Structure

```
training/
├── config.py           # Configuration dataclasses
├── dataset.py          # On-the-fly FPGAN dataset
├── models.py           # Modified ResNet-18 + CosFace
├── trainer.py          # Training loop with DDP support
├── evaluate.py         # Verification evaluation on N2N
├── generate_samples.py # Optional: save samples to disk
└── README.md           # This file

kaggle_train.py         # Main entry point for Kaggle
```

## Configuration

All settings are in `config.py`. Key parameters:

```python
# Data
num_identities = 50000      # Number of synthetic fingerprint IDs
images_per_identity = 11    # Impressions per ID
enable_pseudo_mix = True    # Enable dual augmentation branches
mix_ratio = 0.5             # 50% default, 50% strong augmentation

# Model
embedding_size = 512        # Embedding dimension
remove_first_maxpool = True # Key modification from paper

# CosFace
scale = 64.0               # Scaling factor
margin = 0.35              # Cosine margin

# Training
batch_size = 128           # Per GPU
epochs = 30
lr = 0.1
lr_milestones = [10, 20, 25]
use_amp = True             # Mixed precision training
```

## Augmentation Branches

### Default Branch
- Rotation: ±15°
- Translation: ±8%
- Scale: 95-105%

### Strong Branch
- Rotation: ±30°
- Translation: ±15%
- Scale: 85-115%
- Gaussian blur: 30% probability
- Gaussian noise: σ=0.02

## Evaluation

Evaluate on NIST SD302 (N2N) test set:

```bash
python training/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --n2n-path /path/to/n2n_test \
    --output results/evaluation.json
```

Metrics reported:
- **TAR @ FAR=0.1%**: True Accept Rate at 0.1% False Accept Rate
- **TAR @ FAR=0.01%**: True Accept Rate at 0.01% False Accept Rate
- **EER**: Equal Error Rate

## Memory Requirements

For 2x T4 (16GB each):
- Batch size 128 per GPU works well
- Mixed precision (AMP) reduces memory usage
- On-the-fly generation uses ~2GB for generator model

## Tips for Kaggle

1. **Use GPU accelerator**: Select "GPU T4 x2" in notebook settings
2. **Enable internet**: Required for pip installs
3. **Save outputs**: Checkpoints are saved to `/kaggle/working/checkpoints`
4. **Long training**: Consider saving checkpoints frequently (every 5 epochs)

## Expected Results

With 50,000 identities and 30 epochs, expect:
- Training accuracy: ~95%+
- TAR @ FAR=0.1%: Check paper for benchmarks
- Training time: ~10-15 hours on 2x T4

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size` (try 64 or 96)
- Ensure `use_amp = True`

### Slow training
- Check that both GPUs are being used
- `num_workers=0` is required for on-the-fly CUDA generation

### Generator initialization fails
- Verify model path is correct
- Check that checkpoint files exist in `model_dir/checkpoint/`

## Citation

If you use this code, please cite:

```bibtex
@InProceedings{Shoshan_2024_WACV,
    author    = {Shoshan, Alon and others},
    title     = {FPGAN-Control: A Controllable Fingerprint Generator},
    booktitle = {WACV},
    year      = {2024},
}
```
