# tms_efield_prediction/data/data_splitter.py
import torch
from torch.utils.data import TensorDataset, random_split
import logging

logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, train_pct=0.6, val_pct=0.2, test_pct=0.2, random_seed=42):
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.random_seed = random_seed

    def split_data(self, features, targets):
        """Splits data into training, validation, and test sets."""
        # Transform features from [N, D, H, W, C] (channels-last) to [N, C, D, H, W] (channels-first) for PyTorch Conv3D
        if features.shape[-1] < features.shape[1] : # Basic check if channels seem last
            logger.info("Permuting features from Channels Last -> Channels First")
            features_transformed = features.permute(0, 4, 1, 2, 3)
        else:
            logger.warning(f"Input features shape {features.shape} does not look like channels-last. Assuming channels-first.")
            features_transformed = features # Assume already channels-first

        # Targets are expected as [N, 3, D, H, W] (vectors, channels-first) - no permute needed
        targets_vectors = targets

        logger.info(f"Features shape for Dataset: {features_transformed.shape}")
        logger.info(f"Targets shape for Dataset: {targets_vectors.shape}")

        # Create dataset (Features are Channels First, Targets are Vectors)
        dataset = TensorDataset(features_transformed, targets_vectors)

        # Calculate split sizes
        n_samples = len(dataset)
        n_train = int(n_samples * self.train_pct)
        n_val = int(n_samples * self.val_pct)
        # Adjust test size to ensure total is n_samples
        n_test = n_samples - n_train - n_val
        if n_train + n_val + n_test != n_samples:
            logger.warning(f"Split sizes don't sum perfectly ({n_train}+{n_val}+{n_test}!={n_samples}). Adjusting train size.")
            n_train = n_samples - n_val - n_test # Adjust train size


        logger.info(f"Calculated split sizes: Train={n_train}, Val={n_val}, Test={n_test} (Total={n_samples})")

        # Split dataset
        if n_train <= 0 or n_val < 0 or n_test <= 0:
            raise ValueError(f"Invalid split sizes: Train={n_train}, Val={n_val}, Test={n_test}. Check percentages and data size.")

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.random_seed)  # For reproducibility
        )
        return train_dataset, val_dataset, test_dataset