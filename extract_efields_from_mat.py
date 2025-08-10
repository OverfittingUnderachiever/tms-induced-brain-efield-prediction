import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm

def extract_and_save_efields(mat_file_path, output_dir, subject_id, explore_only=False):
    """Extract E-fields from a .mat file and save them in the expected format"""
    print(f"Processing file: {mat_file_path}")
    
    # Open the file in read mode
    with h5py.File(mat_file_path, 'r') as f:
        # Explore file structure
        print("Top-level keys:")
        for key in f.keys():
            print(f"- {key}")
        
        if explore_only:
            f.visititems(print_attrs)
            return
        
        # Extract E-fields
        print("\nExtracting E-fields...")
        
        # Based on the output, we found the 'efields' key directly
        if 'efields' in f:
            print("Found 'efields' key directly")
            efields = np.array(f['efields'][:])
            print(f"Extracted E-fields with shape: {efields.shape}")
        else:
            # Try other methods to extract E-fields if needed
            print("Could not find 'efields' key. Exploring further...")
            f.visititems(print_attrs)
            return
        
        # Create output directory structure
        multi_sim_dir = os.path.join(output_dir, "multi_sim_100")
        os.makedirs(multi_sim_dir, exist_ok=True)
        
        # Create mesh_outputs directory (needed by the pipeline)
        mesh_outputs_dir = os.path.join(multi_sim_dir, "mesh_outputs")
        os.makedirs(mesh_outputs_dir, exist_ok=True)
        
        # Check for mesh file
        mesh_file = os.path.join(mesh_outputs_dir, f"{subject_id}_efield_first.msh")
        if not os.path.exists(mesh_file):
            print(f"\nNOTE: The voxelization pipeline will expect a mesh file at:")
            print(f"  {mesh_file}")
            print("You will need to copy your mesh file to this location.")
        
        # Save the combined E-fields
        output_path = os.path.join(multi_sim_dir, f"{subject_id}_all_efields.npy")
        np.save(output_path, efields)
        print(f"\nSaved all E-fields to: {output_path}")
        print(f"Final shape: {efields.shape}")
        
        # Optionally save individual E-fields
        individual_dir = os.path.join(multi_sim_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for i in tqdm(range(efields.shape[0]), desc="Saving individual E-fields"):
            ind_path = os.path.join(individual_dir, f"{subject_id}_efield_{i:03d}.npy")
            np.save(ind_path, efields[i])
        
        print(f"Saved {efields.shape[0]} individual E-fields to: {individual_dir}")
        
        # Print a summary of the extracted data
        print("\nE-field extraction summary:")
        print(f"Source: {mat_file_path}")
        print(f"Number of positions: {efields.shape[0]}")
        print(f"Number of nodes per position: {efields.shape[1]}")
        print(f"Format: Scalar E-field magnitudes")
        print(f"Output file: {output_path}")
        print("\nYou can now run the voxelization pipeline with:")
        print(f"python generate_training_data_cli.py --subject {subject_id} --bin_size 25 --formats torch,numpy")
        
        # Explain about the mesh requirement
        print("\nIMPORTANT: For voxelization, you need:")
        print(f"1. The saved E-field data at: {output_path}")
        print(f"2. The mesh file at: {mesh_file}")
        print(f"3. The correct orientation data in: {os.path.join(output_dir, 'orientation_data.json')}")
        print("\nThe mesh file provides the spatial coordinates needed to map the E-field values to voxels.")

def print_attrs(name, obj):
    """Function to print attributes of an HDF5 object"""
    print(f"\nObject: {name}")
    print(f"  Type: {type(obj)}")
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}")
        print(f"  Data type: {obj.dtype}")
        try:
            if len(obj.shape) > 0 and obj.shape[0] > 0:
                if len(obj.shape) == 1:
                    sample_size = min(3, obj.shape[0])
                    print(f"  Sample: {obj[:sample_size]}")
                else:
                    print(f"  First row sample: {obj[0][:5]}")  # Show first 5 values
        except:
            print("  Cannot display sample")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract E-fields from a .mat file")
    parser.add_argument("subject", help="Subject ID (e.g., 006 or sub-006)")
    parser.add_argument("--explore", action="store_true",
                       help="Only explore the file structure without extracting")
    args = parser.parse_args()
    
    # Format subject ID with 'sub-' prefix if needed
    subject_id = args.subject if args.subject.startswith("sub-") else f"sub-{args.subject}"
    
    # Derive paths based on subject ID
    data_root = "/home/freyhe/MA_Henry/data"
    base_dir = os.path.join(data_root, subject_id)
    output_dir = os.path.join(base_dir, "experiment")
    mat_file_path = os.path.join(output_dir, "all", f"{subject_id}_middle_gray_matter_efields.mat")
    
    extract_and_save_efields(mat_file_path, output_dir, subject_id, args.explore)