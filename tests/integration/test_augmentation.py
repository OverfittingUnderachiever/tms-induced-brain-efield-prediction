#!/usr/bin/env python3
"""
Simple Augmentation Test

A minimal script to test if the basic augmentation methods are working.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Try to import BatchAugmentation
try:
    from tms_efield_prediction.data.transformations.augmentation import BatchAugmentation
    print("Successfully imported BatchAugmentation from transformations")
except ImportError:
    try:
        from tms_efield_prediction.data.pipeline.augmentation import BatchAugmentation
        print("Successfully imported BatchAugmentation from pipeline")
    except ImportError:
        print("ERROR: Could not import BatchAugmentation")
        sys.exit(1)

def create_test_data(size=25, batch_size=2):
    """Create simple test data."""
    # Create a batch of 3D tensors with channels
    features = torch.zeros((batch_size, 4, size, size, size), dtype=torch.float32)
    targets = torch.zeros((batch_size, 1, size, size, size), dtype=torch.float32)
    
    # Add some simple patterns
    for b in range(batch_size):
        # Create a sphere in the center
        center = size // 2
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # Simple sphere
                    dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                    if dist < size/4:
                        features[b, 0, i, j, k] = 1.0  # MRI channel
                        features[b, 1:4, i, j, k] = 0.5  # dA/dt channels
                        if dist < size/8:
                            targets[b, 0, i, j, k] = 1.0  # E-field magnitude
    
    return features, targets

def visualize_slice(tensor, slice_idx=None, title=None):
    """Visualize a slice from a tensor."""
    if tensor.ndim == 5:  # [B, C, D, H, W]
        # Take first batch and channel
        tensor = tensor[0, 0]
    elif tensor.ndim == 4:  # [C, D, H, W]
        # Take first channel
        tensor = tensor[0]
    
    # Choose middle slice if not specified
    if slice_idx is None:
        slice_idx = tensor.shape[1] // 2
    
    # Get the slice
    slice_data = tensor[:, slice_idx].cpu().numpy()
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data, cmap='viridis')
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def test_spatial_shift():
    """Test the spatial shift augmentation."""
    print("\n--- Testing Spatial Shift ---")
    
    # Create test data
    features, targets = create_test_data(size=25, batch_size=2)
    
    # Create shifts (use Python integers)
    shifts = torch.zeros((2, 3), dtype=torch.int)
    shifts[0, 0] = 5  # Shift first sample in x dimension
    shifts[1, 1] = 5  # Shift second sample in y dimension
    
    print(f"Shifts shape: {shifts.shape}, dtype: {shifts.dtype}")
    print(f"Shifts: {shifts}")
    
    try:
        # Apply shift
        shifted_features = BatchAugmentation.batch_spatial_shift(
            features,
            shifts,
            dims_first=True
        )
        
        print("Spatial shift successful!")
        print(f"Original shape: {features.shape}")
        print(f"Shifted shape: {shifted_features.shape}")
        
        # Check if the shift worked by comparing means
        original_mean = features.mean().item()
        shifted_mean = shifted_features.mean().item()
        print(f"Original mean: {original_mean:.4f}")
        print(f"Shifted mean: {shifted_mean:.4f}")
        
        # Save before/after images
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(features[0, 0, :, features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Original")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(shifted_features[0, 0, :, shifted_features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Shifted")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("shift_test.png")
        print("Saved visualization to shift_test.png")
        
        return True
    except Exception as e:
        print(f"Error in spatial shift: {e}")
        return False

def test_intensity_scaling():
    """Test the intensity scaling augmentation."""
    print("\n--- Testing Intensity Scaling ---")
    
    # Create test data
    features, targets = create_test_data(size=25, batch_size=2)
    
    # Create scale factors
    scale_factors = torch.tensor([1.5, 0.8])  # Scale factors for each sample
    
    print(f"Scale factors shape: {scale_factors.shape}, dtype: {scale_factors.dtype}")
    print(f"Scale factors: {scale_factors}")
    
    try:
        # Apply intensity scaling
        scaled_features = BatchAugmentation.batch_intensity_scaling(
            features,
            scale_factors,
            dims_first=True
        )
        
        print("Intensity scaling successful!")
        print(f"Original shape: {features.shape}")
        print(f"Scaled shape: {scaled_features.shape}")
        
        # Check if the scaling worked by comparing means
        original_mean = features.mean().item()
        scaled_mean = scaled_features.mean().item()
        expected_mean = original_mean * ((scale_factors[0] + scale_factors[1]) / 2)
        print(f"Original mean: {original_mean:.4f}")
        print(f"Scaled mean: {scaled_mean:.4f}")
        print(f"Expected mean: {expected_mean:.4f}")
        
        # Save before/after images
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(features[0, 0, :, features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Original")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(scaled_features[0, 0, :, scaled_features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Scaled")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("intensity_test.png")
        print("Saved visualization to intensity_test.png")
        
        return True
    except Exception as e:
        print(f"Error in intensity scaling: {e}")
        return False

def test_gaussian_noise():
    """Test the Gaussian noise augmentation."""
    print("\n--- Testing Gaussian Noise ---")
    
    # Create test data
    features, targets = create_test_data(size=25, batch_size=2)
    
    # Create noise std values
    noise_std = torch.tensor([0.1, 0.2])  # Std values for each sample
    
    print(f"Noise std shape: {noise_std.shape}, dtype: {noise_std.dtype}")
    print(f"Noise std: {noise_std}")
    
    try:
        # Apply Gaussian noise
        noisy_features = BatchAugmentation.batch_gaussian_noise(
            features,
            noise_std,
            dims_first=True
        )
        
        print("Gaussian noise successful!")
        print(f"Original shape: {features.shape}")
        print(f"Noisy shape: {noisy_features.shape}")
        
        # Calculate noise level
        noise = noisy_features - features
        noise_level = noise.std().item()
        expected_level = ((noise_std[0]**2 + noise_std[1]**2) / 2) ** 0.5
        print(f"Measured noise level: {noise_level:.4f}")
        print(f"Expected noise level: {expected_level:.4f}")
        
        # Save before/after images
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(features[0, 0, :, features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Original")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(noisy_features[0, 0, :, noisy_features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Noisy")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("noise_test.png")
        print("Saved visualization to noise_test.png")
        
        return True
    except Exception as e:
        print(f"Error in Gaussian noise: {e}")
        return False

def test_elastic_deformation():
    """Test the elastic deformation augmentation."""
    print("\n--- Testing Elastic Deformation ---")
    
    # Create test data
    features, targets = create_test_data(size=25, batch_size=2)
    
    # Create deformation strengths
    deformation_strengths = torch.tensor([2.0, 3.0])  # Strengths for each sample
    
    print(f"Deformation strengths shape: {deformation_strengths.shape}, dtype: {deformation_strengths.dtype}")
    print(f"Deformation strengths: {deformation_strengths}")
    
    try:
        # Apply elastic deformation
        deformed_features = BatchAugmentation.batch_elastic_deformation(
            features,
            deformation_strengths,
            sigma=4.0,
            dims_first=True
        )
        
        print("Elastic deformation successful!")
        print(f"Original shape: {features.shape}")
        print(f"Deformed shape: {deformed_features.shape}")
        
        # Calculate deformation level
        diff = deformed_features - features
        diff_level = diff.std().item()
        print(f"Deformation level: {diff_level:.4f}")
        
        # Save before/after images
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(features[0, 0, :, features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Original")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(deformed_features[0, 0, :, deformed_features.shape[2]//2].cpu().numpy(), cmap='gray')
        plt.title("Deformed")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("elastic_test.png")
        print("Saved visualization to elastic_test.png")
        
        return True
    except Exception as e:
        print(f"Error in elastic deformation: {e}")
        return False

def main():
    """Run all augmentation tests."""
    print("Running augmentation tests...")
    
    # List of test functions
    tests = [
        test_spatial_shift,
        test_intensity_scaling,
        test_gaussian_noise,
        test_elastic_deformation
    ]
    
    # Run tests
    results = {}
    for test_func in tests:
        test_name = test_func.__name__
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"Test {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n=== Test Results ===")
    all_passed = True
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. See details above.")

if __name__ == "__main__":
    main()