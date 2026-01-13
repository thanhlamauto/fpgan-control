#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

"""
FPGAN-Control Training Module

Train fingerprint recognition models using synthetic data
generated on-the-fly by FPGAN-Control.
"""

from .config import Config, get_default_config
from .models import FingerprintRecognitionModel, ModifiedResNet18, CosFaceLoss, build_model
from .dataset import FPGANWrapper, SyntheticFingerprintDataset, OnTheFlyDataLoader
from .trainer import Trainer, train_single_gpu, train_multi_gpu
from .evaluate import evaluate, extract_embeddings, compute_verification_metrics
from .validation import ValidationManager, SyntheticValidationDataset, FVC2004ValidationDataset

__all__ = [
    'Config',
    'get_default_config',
    'FingerprintRecognitionModel',
    'ModifiedResNet18',
    'CosFaceLoss',
    'build_model',
    'FPGANWrapper',
    'SyntheticFingerprintDataset',
    'OnTheFlyDataLoader',
    'Trainer',
    'train_single_gpu',
    'train_multi_gpu',
    'evaluate',
    'extract_embeddings',
    'compute_verification_metrics',
    'ValidationManager',
    'SyntheticValidationDataset',
    'FVC2004ValidationDataset',
]
