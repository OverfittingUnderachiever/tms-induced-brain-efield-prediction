#!/bin/bash
# TMS E-field Prediction Environment Setup
# This script handles all the environment setup automatically

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Setting up TMS E-field Prediction environment..."

# Activate conda environment
if conda activate simnibs3 2>/dev/null; then
    echo "✅ Conda environment 'simnibs3' activated"
else
    echo "❌ Failed to activate conda environment 'simnibs3'"
    echo "   Make sure you've created it with: conda env create -f simnibs3_environment.yml"
    return 1
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "✅ PYTHONPATH configured to: $PROJECT_ROOT"

# Change to the scripts directory
cd "$SCRIPT_DIR"
echo "✅ Working directory: $(pwd)"

# Verify setup
if python -c "import Code.utils; import simnibs" 2>/dev/null; then
    echo "✅ All imports successful - ready to run experiments!"
    echo ""
    echo "📋 Example commands:"
    echo "   python generate_training_data_cli.py --subjects 002 003 004 --bin_size 15 --processes 4 --mri_mode dti"
    echo "   python train_automl_BO.py --num_trials 10"
    echo ""
    echo "💡 Tip: You can now run any script from this directory"
else
    echo "❌ Setup verification failed"
    echo "   Check that SimNIBS is properly installed and all dependencies are available"
    return 1
fi
