TMS E-field Prediction Project

This project is about Transcranial magnetic stimulation specifically the problem of predicting the electric field in the brain based on the magnet position and orientation.
In this project we simulate to generate training test pair using simnibs.
These are then used to train a neural network.

There is a coding convention which should be followed when writing code


here is the file structure
- __init__.py
- venv
  - include
  - lib
    - python3.8
      - site-packages
  - bin
- .pytest_cache
  - v
    - cache
- data
  - __init__.py
  - __pycache__
  - pipeline
    - __init__.py
    - validator.py
    - __pycache__
    - roi_processor.py
    - controller.py
    - loader.py
    - tms_data_types.py
  - formats
    - __init__.py
    - __pycache__
    - simnibs_io.py
  - transformations
    - __init__.py
    - mesh_to_grid.py
    - __pycache__
    - complete_pipeline.py
    - stack_pipeline.py
- __pycache__
- run_tms_simulation.py
- docker

- tests
  - integration
    - test_simulation_integration.py
    - test_tms_pipeline.py
  - unit
    - test_run_simulation.py
    - simnibs_env
      - var
        - cache
          - fontconfig
    - test_runner.py
    - test_matsimnibs_loader.py
    - test_core_infrastructure.py
    - test_tms_loader_paths.py
    - __pycache__
    - test_transformations.py
    - test_simulation_components.py
    - test_roi_processor.py
    - test_full_pipeline.py
    - test_calc_dadt.py
    - test_tms_data_loader.py
    - test_path_resolution.py
    - testing.py
- tms_runner.py
- simulation
  - __init__.py
  - __pycache__
  - pipeline_integration.py
  - coil_position.py
  - tms_simulation.py
  - field_calculation.py
  - runner.py
- utils
  - __init__.py
  - __pycache__
  - pipeline
    - __init__.py
    - __pycache__
    - implementation_unit.py
  - resource
    - __init__.py
    - __pycache__
    - monitor.py
  - debug
    - __init__.py
    - context.py
    - hooks.py
    - __pycache__
    - history.py
  - visualization-utility.py
  - state
    - __init__.py
    - context.py
    - __pycache__
    - transitions.py
---

### **tms_efield_prediction/data/simnibs_io.py**  
**Purpose:**  
Provides utility functions for loading and processing mesh and related data formats used in SimNIBS, a tool for TMS (Transcranial Magnetic Stimulation) E-field prediction.  

**Functionality:**  
- Defines the `MeshData` dataclass to encapsulate mesh structure and metadata.  
- Implements functions for:  
  - Loading SimNIBS mesh files (`load_mesh`).  
  - Loading dA/dt data (`load_dadt_data`).  
  - Loading coil position matrices (`load_matsimnibs`).  
- Supports debugging and resource monitoring via optional hooks (`DebugHook` and `ResourceMonitor`).  

**Key Elements:**  
- `MeshData`: Dataclass for storing mesh structure and metadata.  
- `load_mesh()`: Loads SimNIBS mesh files.  
- `load_dadt_data()`: Loads dA/dt data.  
- `load_matsimnibs()`: Loads coil position matrices.  

**Dependencies & Interactions:**  
- Uses `numpy`, `h5py`, and `simnibs.mesh_io`.  
- Interacts with other components of the `tms_efield_prediction` package.  

**Usage Context:**  
Handles mesh data extraction and preprocessing, ensuring compatibility with SimNIBS workflows.  
Potential errors: file format mismatches, missing datasets, invalid mesh structures. Exception logging and debugging support are included.  

---

### **tms_efield_prediction/data/pipeline/controller.py**  
**Purpose:**  
Manages the execution of the preprocessing pipeline for TMS E-field prediction.  

**Functionality:**  
- Orchestrates data loading, preprocessing, validation, and resource monitoring.  
- Maintains execution logs.  

**Key Elements:**  
- `PipelineExecutionResult`: Stores execution results, including success status, runtime, and error messages.  
- `PipelineController`:  
  - Initializes and executes the pipeline with memory management, debug logging, and validation.  
  - Methods:  
    - `execute_preprocessing()`: Manages end-to-end preprocessing flow.  
    - `_log_execution_result()`: Logs execution details to a JSON file.  

**Dependencies & Interactions:**  
- Uses `numpy`, `torch`.  
- Interacts with internal modules such as `DataLoader`, `Preprocessor`, and `DataValidator`.  

**Usage Context:**  
Handles preprocessing within the overall pipeline execution.  
Potential errors: data loading failures, memory constraints, validation issues.  

---

### **tms_efield_prediction/data/pipeline/loader.py**  
**Purpose:**  
Loads TMS E-field prediction data, integrating MRI, dA/dt, and E-field measurements from structured subject directories.  

**Functionality:**  
- Determines appropriate data file paths, supporting multiple directory structures.  
- Loads essential files, including mesh data, ROI centers, coil positions, and E-field data.  

**Key Elements:**  
- `TMSDataLoader`:  
  - Initializes with `TMSPipelineContext`, an optional `DebugHook`, and `ResourceMonitor`.  
  - `_derive_paths()`: Dynamically sets up file locations based on subject ID.  
  - `load_raw_data()`: Extracts data, leveraging `simnibs_io` and `h5py`.  

**Dependencies & Interactions:**  
- Uses `simnibs_io`, `h5py`.  
- Interacts with multiple components of `tms_efield_prediction`.  

**Usage Context:**  
Ensures smooth data integration in the pipeline.  
Potential errors: missing or incorrectly formatted files, dependency failures, unexpected directory structures.  

---

### **tms_efield_prediction/data/pipeline/roi_processor.py**  
**Purpose:**  
Generates and manages region-of-interest (ROI) meshes as a preprocessing step for TMS E-field simulations.  

**Functionality:**  
- Ensures necessary mesh files exist and verifies input data integrity.  
- Performs computations such as extracting skin normal vectors and defining cylindrical ROI masks.  

**Key Elements:**  
- `ROIProcessor`: Orchestrates mesh processing.  
- `ROIProcessingResult`: Stores processing results.  

**Dependencies & Interactions:**  
- Uses `simnibs.mesh_io` for mesh handling.  
- Relies on `PipelineContext` for state management, `DebugHook` for debugging, `ResourceMonitor` for memory optimization.  

**Usage Context:**  
Prepares ROI meshes for simulations.  
Potential errors: missing/corrupted mesh files, incorrect ROI center specifications, computational errors in ROI mask definition.  

---

### **tms_efield_prediction/data/pipeline/tms_data_types.py**  
**Purpose:**  
Defines structured data representations for handling and processing TMS-related data in a standardized pipeline.  

**Functionality:**  
- Introduces multiple dataclasses and an enumeration for structured data storage:  
  - `TMSRawData`: Stores initial MRI, coil position, and E-field ground truth data.  
  - `TMSProcessedData`: Holds preprocessed features and target values for model input.  
  - `TMSSample`: Represents data for a single coil position.  
  - `TMSSplit`: Organizes dataset splits (training, validation, testing).  
  - `TMSDataType`: Enum defining different data types in the pipeline.  

**Dependencies & Interactions:**  
- Uses `dataclasses`, `numpy`, `typing`, `enum`.  
- Likely integrates with simulation tools like SimNIBS.  

**Usage Context:**  
Standardizes data handling in TMS modeling pipelines, ensuring consistency across ingestion, preprocessing, and training.  
Potential errors: missing/incompatible data formats, requiring validation mechanisms.  

---

### **tms_efield_prediction/data/pipeline/validator.py**  
**Purpose:**  
Validates processed TMS data before further usage in the pipeline.  

**Functionality:**  
- Ensures required files exist, data structures are correct, and values meet expected constraints.  
- Performs numerical and pipeline mode-specific validations.  

**Key Elements:**  
- `DataValidator`: Checks processed TMS data directories.  
- `ValidationError`, `ValidationResult`: Store validation outcomes.  

**Dependencies & Interactions:**  
- Uses `numpy`, `os`.  
- Integrates with `PipelineContext` and `PipelineDebugHook`.  

**Usage Context:**  
Prevents corrupt or incomplete data from propagating through the pipeline.  
Potential errors: missing data files, invalid array shapes, unexpected numerical values.  

---

### **tms_efield_prediction/data/transformations/complete_pipeline.py**  
**Purpose:**  
Defines `CompletePreprocessingPipeline`, integrating mesh-to-grid transformations and channel stacking for TMS E-field data preprocessing.  

**Functionality:**  
- Processes raw TMS data into structured inputs for modeling.  
- Saves processed results for downstream tasks.  

**Dependencies & Interactions:**  
- Uses mesh-to-grid transformation (`MeshToGridTransformer`).  

**Usage Context:**  
Central component of the TMS E-field prediction pipeline.  

---

## tms_efield_prediction/data/transformations/mesh_to_grid.py

### **Purpose:**  
Transforms **mesh-based data** into a **structured 3D grid format**, crucial for **electromagnetic field data processing**. Incorporates debugging and resource monitoring.

### **Functionality:**  
- Converts **unstructured** simulation outputs into **structured grids**.  
- Ensures **efficient transformation** while integrating **pipeline-based execution**.

### **Key Elements:**  
- `MeshToGridTransformer`: Core class handling:
  - **`create_grid`**: Constructs the structured grid.
  - **`voxelize_data`**: Converts mesh data into voxels.
  - **`generate_mask`**: Creates a binary mask for valid voxels.
  - **`create_pipeline`**: Builds the transformation pipeline.
  - **`transform`**: End-to-end transformation execution.

### **Dependencies & Interactions:**  
- **NumPy** for numerical operations.
- **Pipeline utilities**: `ImplementationUnit`, `PipelineDebugHook`, `ResourceMonitor`.

### **Usage Context:**  
- Plays a **key role in data preprocessing** for simulation outputs.
- **Potential Issues:** High memory usage, numerical instability when defining grid boundaries.

---

## tms_efield_prediction/data/transformations/stack_pipeline.py

### **Purpose:**  
Implements a **channel stacking pipeline** for **MRI and dA/dt data** in **TMS E-field prediction**.

### **Functionality:**  
- **Normalizes, scales, and stacks input data channels** for further processing.
- Ensures **correct input format** for model inference.

### **Key Elements:**  
- `StackingConfig`: Defines **normalization methods, scaling factors, output shapes**.
- `ChannelStackingPipeline`: Core pipeline handling:
  - **`_normalize_mri`**: MRI data normalization.
  - **`_normalize_dadt`**: dA/dt data normalization.
  - **`_stack_channels`**: Combines and formats channels.
  - **`process_sample(sample)`**: Processes a single sample.
  - **`process_batch(samples)`**: Batch processing with error handling.

### **Dependencies & Interactions:**  
- **`ImplementationUnit`** for modular transformations.
- **Debugging (`PipelineDebugHook`) & Resource monitoring (`ResourceMonitor`)**.
- **TMS context & data structures**: `TMSPipelineContext`, `TMSSample`, `TMSProcessedData`.

### **Usage Context:**  
- **Preprocessing step** ensuring formatted data for model inference.
- **Potential Issues:** Missing input data, incorrect normalization, batch processing failures.

---

## tms_efield_prediction/simulation/coil_position.py

### **Purpose:**  
Computes **optimal TMS coil positions** on a subject’s head, considering spatial constraints and rotation angles.

### **Functionality:**  
- Aligns coils with **surface normals**.
- Computes **rotation matrices**.
- Integrates with **SimNIBS simulations**.

### **Key Elements:**  
- `CoilPositioningConfig`: Defines **search parameters**.
- `CoilPositionGenerator`: Handles **coil placement calculations**.

### **Dependencies & Interactions:**  
- Uses **`SimulationContext`**, **`DebugHook`**, and **`ResourceMonitor`**.

### **Usage Context:**  
- **Optimizes coil positioning** for accurate simulations.
- **Potential Issues:** Misalignment in rotations, unexpected mesh properties.

---

## tms_efield_prediction/simulation/field_calculation.py

### **Purpose:**  
Calculates **TMS-induced vector fields** with **resource monitoring**.

### **Functionality:**  
- Computes **time derivative of vector potential (dA/dt)** at target positions.
- Supports **parallel computation** and **FMM acceleration (if available)**.

### **Key Elements:**  
- `FieldCalculationConfig`: Defines **precision, parallelization, FMM usage**.
- `FieldCalculator`: Core class handling:
  - **Coil transformation matrices**.
  - **Dipole property computations**.
  - **Parallel processing via `joblib`**.

### **Dependencies & Interactions:**  
- Uses **NumPy, h5py, tqdm, fmm3dpy, simnibs.simulation**.

### **Usage Context:**  
- **Simulation field computation**.
- **Potential Issues:** Missing dependencies, incorrect transformations, memory overload.

---

## tms_efield_prediction/simulation/pipeline_integration.py

### **Purpose:**  
Integrates **TMS simulation** with the **data pipeline**, ensuring structured **data flow and phase isolation**.

### **Functionality:**  
- Manages **simulation execution, preprocessing, and memory optimization**.

### **Key Elements:**  
- `SimulationPipelineAdapter`: Handles **simulation data management**.
- `SimulationPipelineConfig`: Defines **configurations**.

### **Dependencies & Interactions:**  
- Uses **DebugHook, ResourceMonitor, run_simulation, load_mesh_and_roi**.
- Interfaces with **TMSPipelineContext**.

### **Usage Context:**  
- **Ensures seamless data processing** in the pipeline.
- **Potential Issues:** Missing files, misconfigurations, unhandled failures.

---

## tms_efield_prediction/simulation/runner.py

### **Purpose:**  
Manages execution of **TMS E-field simulations**, including **state tracking, resource monitoring, and debugging**.

### **Functionality:**  
- Configures and runs **parallel simulations**.
- Handles **path preparation, data loading, and memory optimization**.
- Supports **debugging and state transitions**.

### **Key Elements:**  
- `SimulationRunnerConfig`: Stores **simulation parameters and paths**.
- `SimulationRunner`: Core execution handler with:
  - **`prepare_paths`**: Manages file paths.
  - **`load_data`**: Loads necessary input data.
  - **`_reduce_memory`**: Optimizes memory usage.

### **Dependencies & Interactions:**  
- Uses **NumPy, h5py, joblib, tqdm**.
- Integrates **TMS-specific modules** for simulations.

### **Usage Context:**  
- **Acts as the controller** in TMS simulation pipelines.
- **Potential Issues:** File errors, memory overload, simulation failures.

---

## tms_efield_prediction/simulation/tms_simulation.py

### **Purpose:**  
Core component of **TMS simulation pipeline** using **SimNIBS**, managing **mesh processing, coil positioning, and field computations**.

### **Functionality:**  
- **Loads and processes mesh data**.
- **Computes coil placements**.
- **Runs electromagnetic field simulations**.

### **Key Elements:**  
- `SimulationState`: Tracks **state and configuration validation**.
- `SimulationContext`: Manages **execution context**.
- `load_mesh_and_roi`: Loads **subject-specific mesh and ROIs**.

### **Dependencies & Interactions:**  
- Uses **SimNIBS, NumPy, h5py, DebugHook, ResourceMonitor**.

### **Usage Context:**  
- **Facilitates structured TMS simulations**.
- **Potential Issues:** Incorrect configurations, missing mesh files, simulation failures.

# tms_efield_prediction/utils/debug/hooks.py

## Purpose
Defines debugging hooks for monitoring and recording events within a pipeline system.

## Functionality
- Implements hooks for logging pipeline activity at configurable sampling rates.
- Records pipeline states, events, and errors in a circular buffer for later retrieval.

## Key Elements
- **`DebugHook`**: Base class that determines whether events should be sampled based on a configurable rate.
- **`PipelineDebugHook`**: Extends `DebugHook` to specifically track pipeline-related events, storing them in a circular buffer. Functions include:
  - `record_state()`
  - `record_event()`
  - `record_error()`

## Dependencies & Interactions
- Imports from `.context` (`DebugContext`, `PipelineDebugContext`, `CircularBuffer`, `PipelineDebugState`) for managing debug configurations and storing logs.
- Uses `traceback` and `time` for error handling and timestamping events.

## Usage Context
- Used in a larger pipeline system for debugging execution flows.
- Errors are always recorded, while states and events follow a configurable sampling rate.
- Potential failure points include misconfigured sampling rates or excessive logging leading to performance issues.

---

# tms_efield_prediction/utils/resource/monitor.py

## Purpose
Monitors system resources, particularly memory and CPU usage, and dynamically manages memory allocation for registered components.

## Functionality
- Registers components and monitors memory/CPU usage in a background thread.
- Triggers memory reduction via callbacks when resource thresholds are exceeded.

## Key Elements
- **`MemoryThresholds`**: Defines critical memory usage levels.
- **`ResourceMetrics`**: Stores memory/CPU statistics.
- **`ResourceMonitor`**: Manages resource tracking, component registration, and automated memory reduction.

## Dependencies & Interactions
- Uses `psutil` for system monitoring.
- Uses `threading` for background execution.

## Usage Context
- Ensures stability in memory-intensive applications by preventing excessive resource consumption.
- Potential failure points include callback errors and thread synchronization issues.

---

# tms_efield_prediction/utils/state/context.py

## Purpose
Defines structured context and state management for a TMS E-field prediction pipeline, handling configuration, validation, and phase transitions.

## Functionality
- Stores dependencies, configurations, and execution parameters for different pipeline phases.
- Tracks execution progress, supports versioning, and enables controlled phase transitions.

## Key Elements
- **Context Management**: `ModelContext`, `PipelineContext`, and `TMSPipelineContext`.
- **State Management**: `ModuleState` and `PipelineState`.

## Dependencies & Interactions
- Uses `dataclasses` for structured data handling.
- Uses `os.path.exists` for file validation.

## Usage Context
- Errors may stem from missing dependencies, misconfigured settings, or invalid pipeline transitions.

---

# tms_efield_prediction/utils/state/transitions.py

## Purpose
Enforces valid state transitions in the **TMS E-field prediction pipeline**.

## Functionality
- Defines and validates transitions between phases (e.g., preprocessing → training, training → evaluation).
- Ensures required data (e.g., processed inputs, model checkpoints) exist before transitioning.

## Key Elements
- **`StateTransitionValidator`**: Handles state transition checks.
- **`PipelineState`** (from `context.py`): Stores and validates pipeline phases.

## Dependencies & Interactions
- Imports `PipelineState` from `context.py`.

## Usage Context
- Errors may occur if prerequisites are missing, preventing pipeline progression.

---

# tms_efield_prediction/utils/visualization-utility.py

## Purpose
Provides tools for loading, processing, and interactively visualizing 3D data related to Transcranial Magnetic Stimulation (TMS) E-field prediction.

## Functionality
- Loads `.npz` files containing volumetric data.
- Generates static and interactive slice views of 3D data.
- Logs errors and handles missing data.

## Key Elements
- **`load_data`**: Reads `.npz` files containing MRI scans, masks, metadata, and optional E-field data.
- **`create_orthogonal_views`**: Generates sagittal, coronal, and axial slice views.
- **`create_interactive_slice_viewer`**: Provides real-time navigation through slices using Matplotlib sliders.

## Dependencies & Interactions
- Uses `numpy`, `matplotlib`, and `argparse`.

## Usage Context
- Used in a TMS analysis pipeline for data inspection.
- Errors may arise from file format issues, missing keys in `.npz` files, or incompatible data dimensions.

---

# tms_efield_prediction/run_tms_simulation.py

## Purpose
Defines `CustomSimulationRunner`, a subclass of `SimulationRunner`, which customizes coil positioning and E-field simulation in TMS simulations.

## Functionality
- Enhances simulation accuracy by integrating a separate mesh for coil positioning.
- Manages batch processing and error handling.

## Key Elements
- **`run`**: Orchestrates the simulation pipeline (loading data, verifying meshes, generating coil positions, managing batch processing).
- **`generate_coil_positions`**: Determines optimal coil placement using mesh data.
- **`load_data`**: Reads and verifies mesh and ROI data.
- **`_run_efield_sim_custom`**: Executes E-field simulations using SimNIBS.
- **`extract_save_efield`**: Processes and extracts E-field values from the simulation output.
- **`_save_positions`**: Stores coil positioning matrices and grid data.

## Dependencies & Interactions
- Interacts with `tms_efield_prediction.simulation` and `simnibs.mesh_io` for mesh handling.
- Uses `scipy.io` for `.mat` file processing.

## Usage Context
- Ensures accurate coil positioning and simulation reproducibility.
- Errors may occur due to invalid mesh data, missing files, or simulation misconfigurations.

# tms_efield_prediction/utils/debug/hooks.py

## Purpose
Defines debugging hooks for monitoring and recording events within a pipeline system.

## Functionality
- Implements hooks for logging pipeline activity at configurable sampling rates.
- Records pipeline states, events, and errors in a circular buffer for later retrieval.

## Key Elements
- **`DebugHook`**: Base class that determines whether events should be sampled based on a configurable rate.
- **`PipelineDebugHook`**: Extends `DebugHook` to specifically track pipeline-related events, storing them in a circular buffer. Functions include:
  - `record_state()`
  - `record_event()`
  - `record_error()`

## Dependencies & Interactions
- Imports from `.context` (`DebugContext`, `PipelineDebugContext`, `CircularBuffer`, `PipelineDebugState`) for managing debug configurations and storing logs.
- Uses `traceback` and `time` for error handling and timestamping events.

## Usage Context
- Used in a larger pipeline system for debugging execution flows.
- Errors are always recorded, while states and events follow a configurable sampling rate.
- Potential failure points include misconfigured sampling rates or excessive logging leading to performance issues.

---

# tms_efield_prediction/utils/resource/monitor.py

## Purpose
Monitors system resources, particularly memory and CPU usage, and dynamically manages memory allocation for registered components.

## Functionality
- Registers components and monitors memory/CPU usage in a background thread.
- Triggers memory reduction via callbacks when resource thresholds are exceeded.

## Key Elements
- **`MemoryThresholds`**: Defines critical memory usage levels.
- **`ResourceMetrics`**: Stores memory/CPU statistics.
- **`ResourceMonitor`**: Manages resource tracking, component registration, and automated memory reduction.

## Dependencies & Interactions
- Uses `psutil` for system monitoring.
- Uses `threading` for background execution.

## Usage Context
- Ensures stability in memory-intensive applications by preventing excessive resource consumption.
- Potential failure points include callback errors and thread synchronization issues.

---

# tms_efield_prediction/utils/state/context.py

## Purpose
Defines structured context and state management for a TMS E-field prediction pipeline, handling configuration, validation, and phase transitions.

## Functionality
- Stores dependencies, configurations, and execution parameters for different pipeline phases.
- Tracks execution progress, supports versioning, and enables controlled phase transitions.

## Key Elements
- **Context Management**: `ModelContext`, `PipelineContext`, and `TMSPipelineContext`.
- **State Management**: `ModuleState` and `PipelineState`.

## Dependencies & Interactions
- Uses `dataclasses` for structured data handling.
- Uses `os.path.exists` for file validation.

## Usage Context
- Errors may stem from missing dependencies, misconfigured settings, or invalid pipeline transitions.

---

# tms_efield_prediction/utils/state/transitions.py

## Purpose
Enforces valid state transitions in the **TMS E-field prediction pipeline**.

## Functionality
- Defines and validates transitions between phases (e.g., preprocessing → training, training → evaluation).
- Ensures required data (e.g., processed inputs, model checkpoints) exist before transitioning.

## Key Elements
- **`StateTransitionValidator`**: Handles state transition checks.
- **`PipelineState`** (from `context.py`): Stores and validates pipeline phases.

## Dependencies & Interactions
- Imports `PipelineState` from `context.py`.

## Usage Context
- Errors may occur if prerequisites are missing, preventing pipeline progression.

---

# tms_efield_prediction/utils/visualization-utility.py

## Purpose
Provides tools for loading, processing, and interactively visualizing 3D data related to Transcranial Magnetic Stimulation (TMS) E-field prediction.

## Functionality
- Loads `.npz` files containing volumetric data.
- Generates static and interactive slice views of 3D data.
- Logs errors and handles missing data.

## Key Elements
- **`load_data`**: Reads `.npz` files containing MRI scans, masks, metadata, and optional E-field data.
- **`create_orthogonal_views`**: Generates sagittal, coronal, and axial slice views.
- **`create_interactive_slice_viewer`**: Provides real-time navigation through slices using Matplotlib sliders.

## Dependencies & Interactions
- Uses `numpy`, `matplotlib`, and `argparse`.

## Usage Context
- Used in a TMS analysis pipeline for data inspection.
- Errors may arise from file format issues, missing keys in `.npz` files, or incompatible data dimensions.

---

# tms_efield_prediction/run_tms_simulation.py

## Purpose
Defines `CustomSimulationRunner`, a subclass of `SimulationRunner`, which customizes coil positioning and E-field simulation in TMS simulations.

## Functionality
- Enhances simulation accuracy by integrating a separate mesh for coil positioning.
- Manages batch processing and error handling.

## Key Elements
- **`run`**: Orchestrates the simulation pipeline (loading data, verifying meshes, generating coil positions, managing batch processing).
- **`generate_coil_positions`**: Determines optimal coil placement using mesh data.
- **`load_data`**: Reads and verifies mesh and ROI data.
- **`_run_efield_sim_custom`**: Executes E-field simulations using SimNIBS.
- **`extract_save_efield`**: Processes and extracts E-field values from the simulation output.
- **`_save_positions`**: Stores coil positioning matrices and grid data.

## Dependencies & Interactions
- Interacts with `tms_efield_prediction.simulation` and `simnibs.mesh_io` for mesh handling.
- Uses `scipy.io` for `.mat` file processing.

## Usage Context
- Ensures accurate coil positioning and simulation reproducibility.
- Errors may occur due to invalid mesh data, missing files, or simulation misconfigurations.



