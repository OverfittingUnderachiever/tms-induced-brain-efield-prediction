# --- START OF FILE k_fold_splitter.py ---
"""
K-Fold Cross-Validation implementation for subject-level splitting.
Supports both standard K-Fold and Leave-One-Out (LOO).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
import os
import json

try:
    from sklearn.model_selection import KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    KFold = None
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not found. Standard K-Fold (k < n_subjects) requires it.")

logger = logging.getLogger(__name__)

class KFoldSubjectSplitter:
    """
    Handles k-fold cross-validation splitting at the subject level.

    Determines splitting strategy based on `k` and `n_subjects`:
    - If `k` == `n_subjects`, performs Leave-One-Out (LOO) cross-validation.
    - If `k` < `n_subjects` and `k` > 1, performs standard K-Fold cross-validation.
    - Raises error for invalid `k` values.
    """

    def __init__(self, k: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Initialize the K-Fold subject splitter.

        Args:
            k: Number of folds. If k equals the number of subjects provided
               to `split` or `split_with_test`, LOO will be performed. Must be >= 2.
            shuffle: Whether to shuffle subjects before splitting (standard K-Fold)
                     or affect fold order (LOO). (default: True)
            random_state: Random seed for reproducibility (default: 42)
        """
        if not isinstance(k, int) or k < 2:
             raise ValueError(f"k must be an integer >= 2, but got {k}")
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state
        logger.info(f"Initialized KFoldSubjectSplitter: k={k}, shuffle={shuffle}, random_state={random_state}")

    def _format_subjects(self, subjects: List[str]) -> np.ndarray:
        """Standardizes subject IDs to 3-digit strings and returns sorted unique numpy array."""
        formatted = []
        for subject in subjects:
            if isinstance(subject, str) and subject.startswith('sub-'):
                subject = subject[4:]
            try:
                formatted.append(str(subject).zfill(3))
            except Exception:
                logger.warning(f"Could not format subject '{subject}', skipping.")
        return np.array(sorted(list(set(formatted))))

    def split(self, subjects: List[str]) -> List[Tuple[List[str], List[str]]]:
        """
        Split subjects into k folds (train/validation).

        Args:
            subjects: List of subject IDs for the training/validation pool.

        Returns:
            List of (train_subjects, val_subjects) tuples for each fold.
        """
        subjects_array = self._format_subjects(subjects)
        n_subjects = len(subjects_array)

        if n_subjects == 0:
            logger.error("No valid subjects provided to split.")
            return []
        if self.k > n_subjects:
             raise ValueError(f"k ({self.k}) cannot be greater than the number of subjects ({n_subjects}).")
        if self.k < 2:
             raise ValueError(f"k must be >= 2, but got {self.k}") # Should be caught by init, but double-check

        folds = []
        indices = np.arange(n_subjects)

        # Determine splitting strategy
        if self.k == n_subjects:
            # --- Leave-One-Out (LOO) ---
            logger.info(f"Performing Leave-One-Out cross-validation (k={self.k}, n_subjects={n_subjects}).")
            if self.shuffle:
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(indices) # Shuffle order of folds

            for i in indices: # Iterate through original indices (order might be shuffled)
                val_original_idx = i
                train_original_indices = np.delete(np.arange(n_subjects), val_original_idx)

                train_subjects = subjects_array[train_original_indices].tolist()
                val_subjects = [subjects_array[val_original_idx]]
                folds.append((train_subjects, val_subjects))

        else:
            # --- Standard K-Fold ---
            logger.info(f"Performing standard {self.k}-Fold cross-validation (n_subjects={n_subjects}).")
            if not SKLEARN_AVAILABLE:
                logger.error("scikit-learn is required for standard K-Fold (k < n_subjects). Install with: pip install scikit-learn")
                raise ImportError("scikit-learn not found, needed for standard K-Fold.")

            kf = KFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
            for train_idx, val_idx in kf.split(indices): # Use indices with KFold
                train_subjects = subjects_array[train_idx].tolist()
                val_subjects = subjects_array[val_idx].tolist()
                folds.append((train_subjects, val_subjects))

        logger.info(f"Created {len(folds)} folds.")
        return folds

    def split_with_test(self, subjects: List[str], test_subjects: List[str]) -> List[Tuple[List[str], List[str], List[str]]]:
        """
        Split subjects into k folds (train/validation) and include a fixed test set.

        Args:
            subjects: List of subject IDs for training/validation pool.
            test_subjects: List of subject IDs for the fixed test set.

        Returns:
            List of (train_subjects, val_subjects, test_subjects) tuples.
        """
        formatted_test_subjects = self._format_subjects(test_subjects).tolist()
        train_val_splits = self.split(subjects) # Handles LOO vs K-Fold

        folds_with_test = []
        test_set = set(formatted_test_subjects)
        for train_subjects, val_subjects in train_val_splits:
            # Optional: Check for overlap, though formatting should prevent this if lists are distinct
            if set(train_subjects).intersection(test_set):
                logger.warning(f"Overlap detected between training and test subjects: {set(train_subjects).intersection(test_set)}")
            if set(val_subjects).intersection(test_set):
                 logger.warning(f"Overlap detected between validation and test subjects: {set(val_subjects).intersection(test_set)}")

            folds_with_test.append((train_subjects, val_subjects, formatted_test_subjects))

        return folds_with_test

    def save_fold_assignments(self, folds: List[Tuple[List[str], List[str], List[str]]],
                             output_dir: str):
        """
        Save fold assignments to a JSON file.

        Args:
            folds: List of (train_subjects, val_subjects, test_subjects) tuples.
            output_dir: Directory to save the assignments file.
        """
        os.makedirs(output_dir, exist_ok=True)
        fold_dict = {}
        for i, (train_subjects, val_subjects, test_subjects) in enumerate(folds):
            fold_dict[f"fold_{i+1}"] = {
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects
            }

        output_path = os.path.join(output_dir, "fold_assignments.json")
        try:
            with open(output_path, 'w') as f:
                json.dump(fold_dict, f, indent=4)
            logger.info(f"Saved fold assignments to {output_path}")
        except Exception as e:
             logger.error(f"Failed to save fold assignments to {output_path}: {e}", exc_info=True)

# --- END OF FILE k_fold_splitter.py ---