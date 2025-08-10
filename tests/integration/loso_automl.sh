#!/bin/bash
# loso_automl.sh - Loop through all subjects for LOSO AutoML

echo "Starting LOSO AutoML Training Loop"


# Holdout=6, Test=7, Val=8,9, Train=1,2,3,4,5,10
echo "=== Holdout Subject 6 ==="
python train_automl_CMAES.py --train-subjects "1,2,3,4,5,10" --val-subjects "8,9" --test-subjects "7" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_6" --sigma0 0.5 --use-stacked-arrays

# Holdout=7, Test=8, Val=9,10, Train=1,2,3,4,5,6
echo "=== Holdout Subject 7 ==="
python train_automl_CMAES.py --train-subjects "1,2,3,4,5,6" --val-subjects "9,10" --test-subjects "8" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_7" --sigma0 0.5 --use-stacked-arrays

# Holdout=8, Test=9, Val=10,1, Train=2,3,4,5,6,7
echo "=== Holdout Subject 8 ==="
python train_automl_CMAES.py --train-subjects "2,3,4,5,6,7" --val-subjects "10,1" --test-subjects "9" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_8" --sigma0 0.5 --use-stacked-arrays

# Holdout=9, Test=10, Val=1,2, Train=3,4,5,6,7,8
echo "=== Holdout Subject 9 ==="
python train_automl_CMAES.py --train-subjects "3,4,5,6,7,8" --val-subjects "1,2" --test-subjects "10" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_9" --sigma0 0.5 --use-stacked-arrays

# Holdout=10, Test=1, Val=2,3, Train=4,5,6,7,8,9
echo "=== Holdout Subject 10 ==="
python train_automl_CMAES.py --train-subjects "4,5,6,7,8,9" --val-subjects "2,3" --test-subjects "1" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_10" --sigma0 0.5 --use-stacked-arrays

# Holdout=1, Test=2, Val=3,4, Train=5,6,7,8,9,10
echo "=== Holdout Subject 1 ==="
python train_automl_CMAES.py --train-subjects "5,6,7,8,9,10" --val-subjects "3,4" --test-subjects "2" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_1" --sigma0 0.5 --use-stacked-arrays

# Holdout=2, Test=3, Val=4,5, Train=1,6,7,8,9,10
echo "=== Holdout Subject 2 ==="
python train_automl_CMAES.py --train-subjects "1,6,7,8,9,10" --val-subjects "4,5" --test-subjects "3" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_2" --sigma0 0.5 --use-stacked-arrays

# Holdout=3, Test=4, Val=5,6, Train=1,2,7,8,9,10
echo "=== Holdout Subject 3 ==="
python train_automl_CMAES.py --train-subjects "1,2,7,8,9,10" --val-subjects "5,6" --test-subjects "4" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_3" --sigma0 0.5 --use-stacked-arrays

# Holdout=4, Test=5, Val=6,7, Train=1,2,3,8,9,10
echo "=== Holdout Subject 4 ==="
python train_automl_CMAES.py --train-subjects "1,2,3,8,9,10" --val-subjects "6,7" --test-subjects "5" --num-samples 70 --max-epochs 15 --max-concurrent 8 --data-dir "/home/freyhe/MA_Henry/data" --output-dir "loso_holdout_4" --sigma0 0.5 --use-stacked-arrays

echo "LOSO AutoML Complete!"