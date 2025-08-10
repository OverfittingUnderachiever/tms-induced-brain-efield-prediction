# tms_efield_prediction/data/pipeline/tms_data_types.py
"""
TMS-specific data type definitions.

This module contains dataclasses and type definitions
for TMS E-field prediction data pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union, NamedTuple
import numpy as np
from enum import Enum
import torch  # ADDED


class TMSDataType(Enum):
    """Types of data in TMS pipeline."""
    MRI = "mri"
    DADT = "dadt"
    EFIELD = "efield"
    COIL_POSITION = "coil_position"
    CONDUCTIVITY = "conductivity"
    ROI = "roi"


@dataclass
class TMSRawData:
    """Container for raw TMS data."""
    subject_id: str
    mri_mesh: Optional[Any] = None  # SimNIBS mesh object
    dadt_data: Optional[np.ndarray] = None  # dA/dt values
    efield_data: Optional[np.ndarray] = None  # E-field ground truth
    coil_positions: Optional[np.ndarray] = None  # Coil position matrices
    roi_center: Optional[Dict[str, np.ndarray]] = None  # ROI information
    metadata: Dict[str, Any] = field(default_factory=dict)
    mri_tensor: Optional[torch.Tensor] = None # ADDED


@dataclass
class TMSProcessedData:
    """Container for processed TMS data ready for model input."""
    subject_id: str
    input_features: np.ndarray  # Combined MRI and dA/dt data
    target_efield: Optional[np.ndarray] = None  # Target E-field (for training)
    mask: Optional[np.ndarray] = None  # Mask of valid regions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TMSSample:
    """Individual sample for a single coil position."""
    sample_id: str
    subject_id: str
    coil_position_idx: int
    mri_data: Optional[np.ndarray] = None  # MRI data
    dadt_data: Optional[np.ndarray] = None  # dA/dt values for this position
    efield_data: Optional[np.ndarray] = None  # E-field for this position
    coil_position: Optional[np.ndarray] = None  # 4x4 transformation matrix
    metadata: Dict[str, Any] = field(default_factory=dict)


class TMSSplit(NamedTuple):
    """Dataset split information."""
    training: List[TMSSample]
    validation: List[TMSSample]
    testing: List[TMSSample]