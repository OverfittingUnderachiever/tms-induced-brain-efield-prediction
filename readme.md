# TMS E-field Prediction: Neural Network-Based Electric Field Prediction for Transcranial Magnetic Stimulation

<div align="center">

![TMS E-field Prediction](https://img.shields.io/badge/TMS-E--field%20Prediction-blue)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)
![SimNIBS](https://img.shields.io/badge/SimNIBS-3.2+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*A comprehensive neural network framework for predicting electric field distributions in Transcranial Magnetic Stimulation (TMS) applications*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

This project implements a state-of-the-art machine learning pipeline for predicting TMS-induced electric field distributions using neural networks. The system bridges the gap between computationally expensive electromagnetic simulations and real-time clinical applications, enabling rapid and accurate E-field predictions for personalized TMS treatment planning.

### 🔬 Key Innovation

- **Fast Prediction**: Reduces computation time from hours to seconds
- **High Accuracy**: Maintains clinical-grade precision in E-field predictions
- **Multi-Modal Input**: Combines MRI data with dA/dt field information
- **Subject Generalization**: Cross-subject model training and validation
- **Production Ready**: Comprehensive AutoML and optimization framework

---

## ✨ Features

### 🧠 **Neural Network Architectures**
- **U-Net Variants**: Specialized 3D U-Net architectures for biomedical data
- **Dual-Modal Models**: Separate processing paths for MRI and dA/dt data
- **Attention Mechanisms**: CBAM-based attention for improved feature extraction
- **Magnitude & Vector Prediction**: Support for both scalar and vector field outputs

<div align="center">
<img src="images/unet_architecture.png" alt="U-Net Architecture" width="700"/>
</div>

*3D U-Net architecture with dynamic feature multipliers and skip connections optimized for TMS field prediction*

### 🔄 **Data Pipeline**
- **SimNIBS Integration**: Seamless workflow with SimNIBS simulation environment
- **Mesh-to-Grid Transformation**: Efficient voxelization of irregular mesh data
- **Field Processing**: Advanced E-field and dA/dt data preprocessing
- **Multi-Subject Support**: Cross-subject training and validation capabilities

### Data Generation & Training Pipeline

<div align="center">
<img src="images/data_flow_diagram.png" alt="TMS Data Pipeline" width="600"/>
</div>

*Complete workflow from SimNIBS mesh processing to voxelized training data generation*

### 🤖 **AutoML & Optimization**
- **Bayesian Optimization**: Automated hyperparameter tuning with Ray Tune
- **CMA-ES Support**: Evolutionary optimization strategies
- **K-Fold Cross-Validation**: Robust model evaluation framework
- **Distributed Training**: Multi-GPU support for large-scale experiments

### Data Augmentation Pipeline

<div align="center">
<img src="images/augmentation_examples.png" alt="Data Augmentation" width="800"/>
</div>

*Comprehensive data augmentation strategies including spatial transformations, elastic deformations, and intensity scaling to improve model generalization*

### 📊 **Visualization & Analysis**
- **3D Field Visualization**: Interactive Three.js-based field rendering
- **Performance Metrics**: Comprehensive evaluation suite for TMS-specific metrics
- **Training Monitoring**: Real-time training progress and resource monitoring
- **Model Comparison**: Side-by-side analysis of different architectures

---

## 🏗️ Architecture

The system follows a **three-tier architecture** with clear separation of concerns:

```mermaid
graph TB
    subgraph "Application Layer"
        CLI[CLI Tools]
        AutoML[AutoML & Experiments]
        Tests[Integration Tests]
    end
    
    subgraph "Infrastructure Layer"
        Utils[Utilities & Visualization]
        Debug[Debug & Monitoring]
        Config[Configuration]
    end
    
    subgraph "Core Domain Layer"
        Models[Neural Networks]
        Pipeline[Data Pipeline]
        Simulation[TMS Simulation]
    end
```

### 🔄 **Data Flow**

```mermaid
graph LR
    A[SimNIBS Mesh] --> B[TMS Simulation]
    B --> C[dA/dt Fields]
    C --> D[E-field Extraction]
    D --> E[Voxelization]
    E --> F[Neural Network]
    F --> G[E-field Prediction]
    
    H[MRI Data] --> E
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.7+** (specifically tested with Python 3.7.16)
- **CUDA-capable GPU** (recommended for training)
- **SimNIBS 3.2+** (for simulation data)
- **Miniconda/Anaconda** (for environment management)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/tms-efield-prediction.git
cd tms-efield-prediction

# 2. Create the conda environment
conda env create -f simnibs3_environment.yml

# 3. Run the setup script (this handles everything!)
source tms_efield_prediction/setup_env.sh
```

**That's it!** You're ready to run experiments. The setup script automatically:
- ✅ Activates the conda environment
- ✅ Sets up all Python paths correctly
- ✅ Verifies everything is working
- ✅ Puts you in the right directory

### Verification

If the setup worked correctly, you should see:
```
🚀 Setting up TMS E-field Prediction environment...
✅ Conda environment 'simnibs3' activated
✅ PYTHONPATH configured
✅ Working directory: /path/to/tms-efield-prediction/tms_efield_prediction
✅ All imports successful - ready to run experiments!
```

### Alternative Setup Options

<details>
<summary>Click to expand alternative installation methods</summary>

#### Manual Environment Setup

```bash
# Create a new conda environment with Python 3.7
conda create -n simnibs3 python=3.7.16

# Activate the environment
conda activate simnibs3

# Install SimNIBS (follow official SimNIBS installation guide)
# This will install most required dependencies

# Install additional packages
pip install torch==1.13.1 torchvision
pip install numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3
pip install pandas==1.3.5 scikit-learn==1.0.2
pip install nibabel h5py pyvista vtk
pip install jupyter ipython
pip install ray optuna bayesian-optimization
pip install monai plotly seaborn bokeh
pip install pytest

# Install the project in development mode
pip install -e .
```

#### Docker Setup

```bash
# Build Docker image
docker build -t tms-efield-prediction .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace tms-efield-prediction
```

</details>

### Important Notes

- **SimNIBS Integration**: This project requires SimNIBS to be properly installed and configured
- **Environment Isolation**: Always use the setup script before running any experiments
- **GPU Support**: CUDA-capable GPU recommended for training, but not required for inference
- **Memory Requirements**: Large models may require 16GB+ RAM for training

---

## ⚡ Quick Start

### Daily Usage

Every time you want to work on the project:

```bash
cd tms-efield-prediction
source tms_efield_prediction/setup_env.sh
```

### Generate Training Data

```bash
# Generate training data for multiple subjects
python generate_training_data_cli.py \
    --subjects 002 003 004 \
    --bin_size 15 \
    --processes 4 \
    --mri_mode dti

# Single subject example
python generate_training_data_cli.py \
    --subjects 001 \
    --bin_size 15 \
    --processes 2 \
    --mri_mode dti
```

### Available Parameters:
- `--subjects`: Subject IDs (space-separated, e.g., `002 003 004`)
- `--bin_size`: Voxel resolution (e.g., `15` for 15mm voxels)
- `--processes`: Number of parallel processes
- `--mri_mode`: MRI processing mode (`dti` recommended)

### Train a Model

```python
from tms_efield_prediction.experiments import MagnitudeExperimentRunner
from tms_efield_prediction.models import SimpleUNetMagnitudeModel

# Configure experiment
config = {
    'model_type': 'unet_magnitude',
    'learning_rate': 1e-3,
    'batch_size': 8,
    'epochs': 100
}

# Run experiment
runner = MagnitudeExperimentRunner(config)
results = runner.train_and_evaluate()
```

### AutoML Optimization

```bash
# Bayesian optimization
python train_automl_BO.py \
    --num_trials 50 \
    --max_epochs 100 \
    --gpus 2

# CMA-ES optimization
python train_automl_CMAES.py \
    --num_trials 30 \
    --population_size 8
```

### Visualize Results

```python
from tms_efield_prediction.utils.visualization import visualize_prediction_vs_ground_truth

# Visualize model predictions
visualize_prediction_vs_ground_truth(
    prediction=pred_field,
    ground_truth=target_field,
    save_path="results/visualization.png"
)
```

---

## 📋 Key Components

### 🧠 **Model Architectures**

| Model | Description | Use Case |
|-------|-------------|----------|
| `SimpleUNetMagnitudeModel` | U-Net for magnitude prediction | Clinical applications requiring scalar fields |
| `SimpleUNetVectorModel` | U-Net for vector field prediction | Research applications requiring directional info |
| `DualModalModel` | Dual-path architecture | Multi-modal input processing |
| `SimpleDualModalModel` | Simplified dual-modal | Robust tensor dimension handling |

### 📊 **Data Pipeline Components**

- **`TMSDataLoader`**: Handles SimNIBS mesh and field data loading
- **`VoxelMapper`**: Efficient mesh-to-grid transformation
- **`FieldProcessor`**: E-field and dA/dt data processing
- **`EnhancedStackingPipeline`**: Multi-channel data preparation

### 🔧 **Training Infrastructure**

- **`ModelTrainer`**: Comprehensive training engine with callbacks
- **`AutoMLConfig`**: Hyperparameter optimization configuration
- **`KFoldSubjectSplitter`**: Subject-level cross-validation
- **`ResourceMonitor`**: GPU and memory management

---

## 📖 Usage Examples

### Training with Custom Architecture

```python
from tms_efield_prediction.models import DualModalModel
from tms_efield_prediction.training import ModelTrainer, TrainerConfig

# Define model configuration
model_config = {
    'input_channels': 4,  # MRI + dA/dt
    'output_channels': 1, # E-field magnitude
    'base_features': 32,
    'depth': 4
}

# Create model and trainer
model = DualModalModel(model_config)
trainer_config = TrainerConfig(
    epochs=100,
    learning_rate=1e-3,
    batch_size=8
)

trainer = ModelTrainer(model, trainer_config)
results = trainer.train(train_loader, val_loader)
```

### Multi-Subject Cross-Validation

```python
from tms_efield_prediction.automl.integration import KFoldAutoMLManager

# Setup K-fold cross-validation
kfold_manager = KFoldAutoMLManager(
    subjects=['sub-01', 'sub-02', 'sub-03', 'sub-04'],
    n_folds=4,
    optimization_method='bayesian'
)

# Run cross-validated optimization
results = kfold_manager.run_kfold_optimization(
    num_trials=20,
    max_epochs=50
)
```

### Advanced Visualization

```python
from tms_efield_prediction.utils.PC_visualization import visualize_point_clouds

# Create 3D visualization
visualize_point_clouds(
    points=[mesh_nodes],
    intensities=[field_magnitudes],
    colors=['field_intensity'],
    output_path='field_visualization.html',
    point_size=2.0
)
```

---

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  architecture: "dual_modal"
  input_channels: 4
  output_channels: 1
  base_features: 32
  depth: 4
  use_attention: true

training:
  learning_rate: 1e-3
  batch_size: 8
  epochs: 100
  optimizer: "adam"
  scheduler: "cosine"

data:
  bin_size: 64
  augmentation: true
  normalization: "z_score"
```

### System Constants (`constants.py`)

```python
# E-field masking threshold
EFIELD_MASK_THRESHOLD = 1e-8

# Default voxel resolution
DEFAULT_VOXEL_SIZE = 1.0  # mm

# GPU memory management
GPU_MEMORY_THRESHOLD = 0.8
```

---

## 📊 Performance Metrics

The system provides comprehensive evaluation metrics tailored for TMS applications:

- **Magnitude Error**: Mean absolute/relative error in field magnitudes
- **Angular Error**: Directional accuracy for vector predictions
- **Hotspot Accuracy**: Clinical relevance of high-intensity regions
- **Field Similarity**: Combined magnitude and directional metrics
- **Correlation Coefficients**: Statistical similarity measures

### Model Performance Comparison

<div align="center">
<img src="images/error_distribution_comparison.png" alt="Model Performance Comparison" width="900"/>
</div>

*Error distribution analysis across different model architectures showing consistent sub-10% MAE performance with ensemble models achieving optimal accuracy*

---

## 🛠️ Development

### Running Tests

```bash
# Use the setup script first
source tms_efield_prediction/setup_env.sh

# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models/
python -m pytest tests/test_pipeline/
python -m pytest tests/integration/
```

### Code Quality

```bash
# Format code
black tms_efield_prediction/

# Lint code
flake8 tms_efield_prediction/

# Type checking
mypy tms_efield_prediction/
```

### Debugging

The system includes comprehensive debugging infrastructure:

```python
from tms_efield_prediction.debug import PipelineDebugHook

# Enable debugging
debug_hook = PipelineDebugHook(sample_rate=0.1)
context.debug_hook = debug_hook

# Debug information is automatically collected
```

---

## 🚨 Troubleshooting

### Common Issues

1. **Setup script fails with "environment not found"**
   ```bash
   # Make sure you created the environment first
   conda env create -f simnibs3_environment.yml
   ```

2. **"No module named 'Code'" error**
   ```bash
   # Make sure you run the setup script from the project root
   cd tms-efield-prediction
   source tms_efield_prediction/setup_env.sh
   ```

3. **SimNIBS import error**
   ```bash
   # Ensure SimNIBS is properly installed in your conda environment
   conda activate simnibs3
   python -c "import simnibs; print('SimNIBS version:', simnibs.__version__)"
   ```

4. **Permission denied**
   ```bash
   # Make sure the setup script is executable
   chmod +x tms_efield_prediction/setup_env.sh
   ```

5. **CUDA/PyTorch Issues**
   ```bash
   # Check PyTorch CUDA compatibility
   python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```

### Pro Tip: Create a Global Alias

For even easier access, add this to your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'alias tms-setup="cd /path/to/tms-efield-prediction && source tms_efield_prediction/setup_env.sh"' >> ~/.bashrc
source ~/.bashrc
```

Then you can just run `tms-setup` from anywhere!

### Getting Help

If you encounter issues:

1. Check that all prerequisites are installed
2. Verify the conda environment was created successfully: `conda env list`
3. Make sure SimNIBS is working: `conda activate simnibs3 && python -c "import simnibs"`
4. Run the setup script with verbose output to see what's failing

The setup script will tell you exactly what went wrong and how to fix it.

---

## 📚 Documentation

- **[System Architecture](docs/architecture.md)**: Detailed technical architecture
- **[API Reference](docs/api/)**: Complete API documentation
- **[User Guide](docs/user_guide.md)**: Step-by-step usage instructions
- **[Development Guide](docs/development.md)**: Contributing and development setup
- **[Research Paper](docs/thesis.pdf)**: Academic foundation and validation

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Set up the environment: `conda env create -f simnibs3_environment.yml`
4. Activate environment: `source tms_efield_prediction/setup_env.sh`
5. Make your changes and add tests
6. Run the test suite: `pytest`
7. Submit a pull request

### Areas for Contribution

- 🧠 New neural network architectures
- 📊 Additional evaluation metrics
- 🔧 Performance optimizations
- 📖 Documentation improvements
- 🐛 Bug fixes and testing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SimNIBS Team**: For the excellent electromagnetic simulation framework
- **PyTorch Community**: For the deep learning infrastructure
- **Ray Team**: For distributed computing and hyperparameter optimization
- **Research Contributors**: For validation and clinical insights

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/tms-efield-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/tms-efield-prediction/discussions)
- **Email**: your.email@domain.com

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for the TMS research community

</div>
