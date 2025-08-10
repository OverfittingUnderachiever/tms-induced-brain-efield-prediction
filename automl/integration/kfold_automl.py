# --- START OF FILE kfold_automl.py ---
"""
K-Fold Cross-Validation integration for AutoML.
Manages the process of running AutoML across different data folds.
"""

import os
import logging
import numpy as np
import json
import yaml
import glob
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import torch
from ray import tune

# Ensure correct relative import path based on your project structure
from ...data.k_fold_splitter import KFoldSubjectSplitter
from ...experiments.tracking import ExperimentTracker
from ...utils.resource.monitor import ResourceMonitor

logger = logging.getLogger(__name__)

class KFoldAutoMLManager:
    """
    Manages k-fold cross-validation for AutoML optimization.
    Relies on KFoldSubjectSplitter for generating folds.
    """

    def __init__(
        self,
        k: int, # Explicitly require k
        base_output_dir: str,
        fold_dir_name: str = "k_fold_cv",
        shuffle: bool = True,
        random_state: int = 42
    ):
        if not isinstance(k, int) or k < 2:
             raise ValueError(f"KFoldAutoMLManager requires k to be an integer >= 2, but got {k}")
        self.k = k
        self.base_output_dir = base_output_dir
        self.fold_dir_name = fold_dir_name
        self.shuffle = shuffle
        self.random_state = random_state

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.kfold_base_dir = os.path.abspath(os.path.join(
            base_output_dir, f"{fold_dir_name}_{self.timestamp}"
        ))
        os.makedirs(self.kfold_base_dir, exist_ok=True)

        self.kfold_splitter = None # Instantiated later
        self.fold_results = []
        logger.info(f"Initialized KFoldAutoMLManager for {k}-fold CV.")

    @classmethod
    def for_leave_one_out(
        cls,
        subjects: List[str],
        base_output_dir: str,
        fold_dir_name: str = "leave_one_out",
        shuffle: bool = False,
        random_state: int = 42
    ):
        k = len(subjects)
        if k < 2:
            raise ValueError(f"Leave-one-out requires at least 2 subjects, but got {k}.")
        logger.info(f"Configuring KFoldAutoMLManager for Leave-One-Out (k={k}).")

        return cls(
            k=k,
            base_output_dir=base_output_dir,
            fold_dir_name=fold_dir_name,
            shuffle=shuffle,
            random_state=random_state
        )

    def check_subject_directories(self, data_root_path: str, subjects: List[str]) -> List[str]:
        valid_subjects = []
        if not os.path.isdir(data_root_path):
            logger.error(f"Data root path does not exist: {data_root_path}")
            return []
        for subject in subjects:
            sub_id = str(subject).strip().zfill(3) # Standardize format
            sub_dir = os.path.join(data_root_path, f"sub-{sub_id}")
            if os.path.isdir(sub_dir):
                 # TODO: Add more robust checks here if needed (e.g., experiment dir, file existence)
                 valid_subjects.append(sub_id) # Store standardized ID
            else:
                 logger.warning(f"Directory not found for subject {subject} (checked as {sub_id}) at {sub_dir}")
        logger.info(f"Checked subjects {subjects} -> Valid subject directories found: {valid_subjects}")
        return valid_subjects


    def prepare_folds(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str]
    ) -> List[Tuple[List[str], List[str], List[str]]]:
        # Combine train and validation subjects into a single pool for splitting
        all_train_val_subjects = sorted(list(set(train_subjects + val_subjects)))

        if not all_train_val_subjects:
            raise ValueError("No subjects provided for training/validation pool.")

        # Check if k needs adjustment
        num_subjects_to_split = len(all_train_val_subjects)
        if self.k > num_subjects_to_split:
             logger.warning(f"Requested k ({self.k}) > available subjects ({num_subjects_to_split}). Adjusting k to {num_subjects_to_split}.")
             self.k = num_subjects_to_split
        elif self.k < 2:
             # This should ideally be caught in __init__ but check again
             logger.error(f"Invalid k value ({self.k}) detected before splitting. Requires k >= 2.")
             raise ValueError(f"Cannot prepare folds with k={self.k}")


        self.kfold_splitter = KFoldSubjectSplitter(
            k=self.k,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        logger.info(f"Preparing {self.k} folds from subjects: {all_train_val_subjects} using KFoldSubjectSplitter.")
        folds = self.kfold_splitter.split_with_test(
            subjects=all_train_val_subjects,
            test_subjects=test_subjects
        )

        if not folds:
             logger.error("KFoldSubjectSplitter returned no folds.")
             return []

        self.kfold_splitter.save_fold_assignments(folds, self.kfold_base_dir)
        fold_type = "Leave-One-Out" if self.k == num_subjects_to_split else f"Standard {self.k}-Fold"
        logger.info(f"Generated {len(folds)} folds using {fold_type} strategy.")
        return folds

    def run_kfold_optimization(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
        optimization_function,
        main_args: Any,
        extra_optimization_args: Optional[Dict[str, Any]] = None,
        max_folds_to_run: Optional[int] = None # <<< ADDED PARAMETER
    ) -> Dict[str, Any]:
        if extra_optimization_args is None:
            extra_optimization_args = {}

        # Optional: Validate subject directories before preparing folds
        # data_root_path = main_args.data_dir
        # train_subjects = self.check_subject_directories(data_root_path, train_subjects)
        # val_subjects = self.check_subject_directories(data_root_path, val_subjects)
        # test_subjects = self.check_subject_directories(data_root_path, test_subjects)
        # if not train_subjects and not val_subjects: # Need at least one subject for train/val pool
        #     raise ValueError("No valid subject directories found for training/validation.")

        folds = self.prepare_folds(train_subjects, val_subjects, test_subjects)

        if not folds:
            logger.error("Fold preparation failed. Aborting optimization.")
            return {"error": "Fold preparation failed."}

        num_folds_generated = len(folds)
        self.fold_results = [] # Reset results for this run

        # Determine how many folds to actually execute
        num_folds_to_execute = num_folds_generated
        if max_folds_to_run is not None and max_folds_to_run > 0:
            if max_folds_to_run < num_folds_generated:
                logger.warning(f"Limiting execution to the first {max_folds_to_run} out of {num_folds_generated} generated folds.")
                num_folds_to_execute = max_folds_to_run
            else:
                logger.info(f"Requested max_folds_to_run ({max_folds_to_run}) >= generated folds ({num_folds_generated}). Running all {num_folds_generated} generated folds.")
        elif max_folds_to_run is not None and max_folds_to_run <= 0:
             logger.warning(f"max_folds_to_run ({max_folds_to_run}) is not positive. Running all {num_folds_generated} generated folds.")


        # Loop through the folds to be executed
        for fold_idx in range(num_folds_to_execute):
            fold_train_subjects, fold_val_subjects, fold_test_subjects = folds[fold_idx]

            logger.info(f"\n{'='*80}\nExecuting fold {fold_idx+1}/{num_folds_to_execute} (out of {num_folds_generated} generated)\n{'='*80}")
            logger.info(f"Train subjects: {fold_train_subjects}")
            logger.info(f"Validation subjects: {fold_val_subjects}")
            logger.info(f"Test subjects: {fold_test_subjects}")

            fold_dir = os.path.join(self.kfold_base_dir, f"fold_{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)

            if len(fold_val_subjects) == 1:
                val_subject = fold_val_subjects[0]
                fold_experiment_name = f"val_{val_subject}_fold_{fold_idx+1}"
            else:
                fold_experiment_name = f"automl_fold_{fold_idx+1}"

            fold_run_args = {
                'train_subjects': fold_train_subjects,
                'val_subjects': fold_val_subjects,
                'test_subjects': fold_test_subjects,
                'args': main_args,
                'output_dir': fold_dir,
                'experiment_name': fold_experiment_name,
                **extra_optimization_args
            }

            try:
                fold_result = optimization_function(**fold_run_args)
                # Expecting: analysis, best_trial, best_config, best_metrics
                analysis, best_trial, best_config, best_metrics = fold_result
                self.fold_results.append({
                    "fold": fold_idx + 1, "status": "success",
                    "train_subjects": fold_train_subjects, "val_subjects": fold_val_subjects, "test_subjects": fold_test_subjects,
                    "best_trial_id": best_trial.trial_id if best_trial else "unknown",
                    "best_config": best_config, "best_metrics": best_metrics
                })
            except Exception as e:
                logger.error(f"Error running optimization for fold {fold_idx+1}: {e}", exc_info=True)
                self.fold_results.append({
                    "fold": fold_idx + 1, "status": "error",
                    "train_subjects": fold_train_subjects, "val_subjects": fold_val_subjects, "test_subjects": fold_test_subjects,
                    "error_message": str(e)
                })

        kfold_results = self._aggregate_results()
        self._save_results(kfold_results)
        self._print_summary(kfold_results)
        return kfold_results

    def _aggregate_results(self) -> Dict[str, Any]:
        aggregate_metrics = {}
        metric_keys = set()
        successful_folds = [res for res in self.fold_results if res.get("status") == "success"]

        if not successful_folds:
            logger.warning("No successful folds to aggregate results from.")
            return {
                "num_folds_requested": getattr(self, 'k', 'N/A'), # Use self.k if available
                "num_folds_run": len(self.fold_results),
                "num_folds_successful": 0, "folds": self.fold_results,
                "aggregate_metrics": {}, "timestamp": self.timestamp
            }

        for fold_result in successful_folds:
            if fold_result.get("best_metrics"):
                metric_keys.update(fold_result["best_metrics"].keys())

        for key in metric_keys:
            metric_values = []
            for fold in successful_folds:
                 metric_val = fold.get("best_metrics", {}).get(key)
                 if isinstance(metric_val, (int, float, np.number)):
                     metric_values.append(metric_val)
                 elif metric_val is not None:
                     logger.debug(f"Metric '{key}' in fold {fold.get('fold')} is non-numeric ({type(metric_val)}), skipping aggregation.")

            if metric_values:
                aggregate_metrics[key] = {
                    "mean": float(np.mean(metric_values)), "std": float(np.std(metric_values)),
                    "min": float(np.min(metric_values)), "max": float(np.max(metric_values)),
                    "count": len(metric_values), "values": metric_values
                }

        kfold_results = {
            "num_folds_requested": getattr(self, 'k', 'N/A'),
            "num_folds_run": len(self.fold_results),
            "num_folds_successful": len(successful_folds),
            "folds": self.fold_results,
            "aggregate_metrics": aggregate_metrics, "timestamp": self.timestamp
        }
        return kfold_results

    def _save_results(self, kfold_results: Dict[str, Any]) -> None:
        results_path_json = os.path.join(self.kfold_base_dir, "kfold_results.json")
        results_path_yaml = os.path.join(self.kfold_base_dir, "kfold_results.yaml")

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                if isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
                if isinstance(obj, (np.bool_)): return bool(obj)
                if isinstance(obj, (np.void)): return None
                return json.JSONEncoder.default(self, obj)

        try:
            with open(results_path_json, 'w') as f:
                json.dump(kfold_results, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Saved k-fold results to JSON: {results_path_json}")
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}", exc_info=True)
        try:
            with open(results_path_yaml, 'w') as f:
                yaml.dump(kfold_results, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved k-fold results to YAML: {results_path_yaml}")
        except Exception as e:
            logger.error(f"Failed to save results to YAML: {e}", exc_info=True)

    def _print_summary(self, kfold_results: Dict[str, Any]) -> None:
        logger.info(f"\n{'='*80}\nK-Fold Cross-Validation Summary\n{'='*80}")
        logger.info(f"Results directory: {self.kfold_base_dir}")
        logger.info(f"Timestamp: {kfold_results.get('timestamp', 'N/A')}")
        logger.info(f"Folds requested (k): {kfold_results.get('num_folds_requested', 'N/A')}")
        logger.info(f"Folds run: {kfold_results.get('num_folds_run', 'N/A')}")
        logger.info(f"Folds successful: {kfold_results.get('num_folds_successful', 'N/A')}")

        if kfold_results.get("aggregate_metrics"):
            logger.info("\nAggregate Metrics (from successful folds):")
            for key, stats in kfold_results["aggregate_metrics"].items():
                 logger.info(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, count: {stats['count']})")
        else:
            logger.info("\nNo aggregate metrics available.")

        error_folds = [f["fold"] for f in kfold_results.get("folds", []) if f.get("status") == "error"]
        if error_folds:
            logger.warning(f"\nErrors occurred in folds: {error_folds}")
        logger.info(f"\nFull results saved in: {self.kfold_base_dir}\n{'='*80}")


def run_automl_with_params(
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    output_dir: str,
    optimization_function,
    main_args: Any,
    extra_optimization_args: Optional[Dict[str, Any]] = None,
    fold_dir_name: str = "kfold_run", # Changed default name
    k_folds: Optional[int] = None,
    max_folds_to_run: Optional[int] = None # <<< ADDED PARAMETER
):
    if main_args is None:
        logger.error("main_args object is required but was not provided.")
        raise ValueError("main_args must be provided to run_automl_with_params")

    # Clean inputs
    train_subjects = [str(s).strip() for s in train_subjects if str(s).strip()]
    if isinstance(val_subjects, list) and len(val_subjects) == 1 and str(val_subjects[0]).lower() == 'loo':
        is_leave_one_out = True
        # Keep val_subjects as ['loo'] for the flag, manager will handle the pool
    else:
        is_leave_one_out = False
        val_subjects = [str(s).strip() for s in val_subjects if str(s).strip()] # Clean regular list
    test_subjects = [str(s).strip() for s in test_subjects if str(s).strip()]

    manager = None # Initialize manager variable

    # --- Leave-One-Out Cross-Validation ---
    if is_leave_one_out:
        logger.info(f"Detected Leave-One-Out request for subjects: {train_subjects}")
        if not train_subjects:
             raise ValueError("Leave-One-Out requires at least one subject in train_subjects.")
        manager = KFoldAutoMLManager.for_leave_one_out(
            subjects=train_subjects,
            base_output_dir=output_dir,
            fold_dir_name=fold_dir_name,
            # shuffle=getattr(main_args, 'shuffle_folds', False), # Example: get from args if needed
            # random_state=getattr(main_args, 'random_seed', 42)
        )
        # For manager's run call, provide original train list and empty val list
        subjects_for_run_train = train_subjects
        subjects_for_run_val = []

    # --- Standard K-Fold Cross-Validation ---
    elif k_folds and isinstance(k_folds, int) and k_folds >= 2:
        logger.info(f"Running standard {k_folds}-Fold cross-validation.")
        all_subjects_pool = sorted(list(set(train_subjects + val_subjects)))
        if not all_subjects_pool:
             raise ValueError("Standard K-Fold requires subjects in train_subjects or val_subjects.")
        manager = KFoldAutoMLManager(
            k=k_folds,
            base_output_dir=output_dir,
            fold_dir_name=f"{k_folds}_fold_cv", # Use k in name
            # shuffle=getattr(main_args, 'shuffle_folds', True),
            # random_state=getattr(main_args, 'random_seed', 42)
        )
        # For manager's run call, provide combined pool as train, empty val
        subjects_for_run_train = all_subjects_pool
        subjects_for_run_val = []

    # --- Standard Single Optimization (No Cross-Validation) ---
    else:
        logger.info("Running standard optimization with fixed train/validation split (no cross-validation).")
        if not train_subjects: raise ValueError("Standard run requires training subjects.")
        if not val_subjects: logger.warning("Standard run: No validation subjects provided.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"automl_single_run_{timestamp}"
        single_run_output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(single_run_output_dir, exist_ok=True)

        standard_run_args = {
            'train_subjects': train_subjects, 'val_subjects': val_subjects, 'test_subjects': test_subjects,
            'args': main_args, 'output_dir': single_run_output_dir,
            'experiment_name': experiment_name, **(extra_optimization_args or {})
        }
        try:
            # Directly call the optimization function once
            return optimization_function(**standard_run_args)
        except Exception as e:
            logger.error(f"Error during single optimization run: {e}", exc_info=True)
            return {"error": f"Single optimization run failed: {e}"}

    # --- Execute K-Fold/LOO run using the configured manager ---
    if manager:
        return manager.run_kfold_optimization(
            train_subjects=subjects_for_run_train,
            val_subjects=subjects_for_run_val,
            test_subjects=test_subjects,
            optimization_function=optimization_function,
            main_args=main_args,
            extra_optimization_args=extra_optimization_args,
            max_folds_to_run=max_folds_to_run # Pass the limit here
        )
    else:
        # Should not happen if logic above is correct, but as a safeguard
        logger.error("Failed to configure manager for LOO or K-Fold.")
        return {"error": "Could not determine run mode (LOO/K-Fold/Single)."}


# --- END OF FILE kfold_automl.py ---