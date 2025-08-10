#!/bin/bash
# run_memory_experiment.sh
# Convenience script to run memory consumption experiments

set -e  # Exit on any error

# Default values
DATA_DIR="/home/freyhe/MA_Henry/data"
OUTPUT_DIR="./memory_experiments"
MEMORY_STRESS="heavy"  # Changed default to heavy for more interesting results
MONITORING_INTERVAL=0.2  # Faster sampling for better spike detection
TRAIN_SUBJECTS="4,6,7"
VAL_SUBJECTS="3"
NUM_GPUS=""
FORCE_SINGLE_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --memory-stress)
            MEMORY_STRESS="$2"
            shift 2
            ;;
        --monitoring-interval)
            MONITORING_INTERVAL="$2"
            shift 2
            ;;
        --train-subjects)
            TRAIN_SUBJECTS="$2"
            shift 2
            ;;
        --val-subjects)
            VAL_SUBJECTS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --force-single-gpu)
            FORCE_SINGLE_GPU=true
            shift
            ;;
        --help|-h)
            echo "Memory Consumption Experiment Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR              Data directory path (default: $DATA_DIR)"
            echo "  --output-dir DIR            Output directory (default: $OUTPUT_DIR)"
            echo "  --memory-stress LEVEL       Memory stress level: light|medium|heavy|extreme (default: $MEMORY_STRESS)"
            echo "  --monitoring-interval SEC   Memory monitoring interval in seconds (default: $MONITORING_INTERVAL)"
            echo "  --train-subjects LIST       Comma-separated training subjects (default: $TRAIN_SUBJECTS)"
            echo "  --val-subjects LIST         Comma-separated validation subjects (default: $VAL_SUBJECTS)"
            echo "  --num-gpus N                Number of GPUs to use (default: all available)"
            echo "  --force-single-gpu          Force single GPU usage"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Memory stress levels:"
            echo "  light:   Small model, low memory usage (batch=64*GPUs, features=32, levels=3)"
            echo "  medium:  Medium model, moderate memory usage (batch=96*GPUs, features=64, levels=4)"
            echo "  heavy:   Large model, high memory usage (batch=96*GPUs, features=128, levels=5) [RTX 2080 Ti optimized]"
            echo "  extreme: Maximum model, extreme memory usage (batch=128*GPUs, features=256, levels=6) [RTX 2080 Ti optimized]"
            echo ""
            echo "Note: Batch sizes are per GPU and will be multiplied by the number of GPUs used."
            echo "NOW MONITORS RESERVED MEMORY - the REAL GPU usage (not just allocated memory)!"
            echo "Heavy/Extreme levels target 80-90% RESERVED memory utilization to avoid OOM."
            echo ""
            echo "Examples:"
            echo "  $0 --memory-stress extreme --num-gpus 4"
            echo "  $0 --memory-stress heavy --force-single-gpu"
            echo "  $0 --data-dir /path/to/data --memory-stress heavy --monitoring-interval 0.1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "==============================================="
echo "Memory Consumption Experiment"
echo "==============================================="
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Memory Stress Level: $MEMORY_STRESS"
echo "Monitoring Interval: ${MONITORING_INTERVAL}s"
echo "Training Subjects: $TRAIN_SUBJECTS"
echo "Validation Subjects: $VAL_SUBJECTS"
if [ -n "$NUM_GPUS" ]; then
    echo "Number of GPUs: $NUM_GPUS"
else
    echo "Number of GPUs: All available"
fi
if [ "$FORCE_SINGLE_GPU" = true ]; then
    echo "Force Single GPU: Yes"
fi
echo "==============================================="

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Python script exists
SCRIPT_PATH="./memory_consumption_experiment.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script $SCRIPT_PATH not found"
    echo "Make sure you're running this from the correct directory"
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU monitoring may not work properly."
fi

# Check Python dependencies
echo "Checking Python environment..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"; then
    python3 -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
fi
echo ""

# Run the experiment
echo "Starting memory consumption experiment..."
echo "Press Ctrl+C to interrupt if needed"
echo ""

# Store the start time
START_TIME=$(date)
echo "Experiment started at: $START_TIME"

# Build the command with optional GPU arguments
CMD_ARGS=(
    "--data-dir" "$DATA_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--memory-stress" "$MEMORY_STRESS"
    "--monitoring-interval" "$MONITORING_INTERVAL"
    "--train-subjects" "$TRAIN_SUBJECTS"
    "--val-subjects" "$VAL_SUBJECTS"
    "--use-stacked-arrays"
)

# Add GPU-specific arguments if specified
if [ -n "$NUM_GPUS" ]; then
    CMD_ARGS+=("--num-gpus" "$NUM_GPUS")
fi

if [ "$FORCE_SINGLE_GPU" = true ]; then
    CMD_ARGS+=("--force-single-gpu")
fi

# Run the Python script with all parameters
python3 "$SCRIPT_PATH" "${CMD_ARGS[@]}"

# Check if the experiment completed successfully
EXPERIMENT_EXIT_CODE=$?
END_TIME=$(date)

echo ""
echo "==============================================="
echo "Experiment completed at: $END_TIME"
echo "Started at: $START_TIME"

if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    echo ""
    echo "Results available in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - Memory usage plots (PNG files)"
    echo "  - Raw memory data (CSV and pickle files)"
    echo "  - Experiment configuration and results (YAML files)"
    echo ""
    echo "To analyze the data further, run:"
    echo "  python3 analyze_memory_data.py [experiment_directory]"
    echo ""
    echo "To view the plots, transfer the PNG files to your local machine:"
    echo "  scp [user@server]:[experiment_directory]/*.png ./local_directory/"
else
    echo "Status: FAILED (exit code: $EXPERIMENT_EXIT_CODE)"
    echo ""
    echo "Check the experiment logs for error details."
fi

echo "==============================================="