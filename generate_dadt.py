import time
import os
import h5py
import numpy as np
from simnibs import mesh_io
from tms_simulation import calc_dAdt

# Process all subjects from 001 to 009

def generate_dadt_from_mat(subject_id):
    number = subject_id.split("-")[1]
    sub_id = number
    print(f"\n{'='*40}\nProcessing subject {sub_id}\n{'='*40}")

    # set file names and parameters
    # ======================================================================
    exp_path = '/home/freyhe/MA_Henry/data/sub-'+sub_id+'/experiment/all/'
    output_path = '/home/freyhe/MA_Henry/data/sub-'+sub_id+'/experiment/dadt_roi_maps/'
    fn_coil = '/home/freyhe/MA_Henry/data/coil/MagVenture_Cool-B65.ccd'
    fn_mesh_roi = exp_path + 'sub-'+sub_id+'_middle_gray_matter_roi.msh'
    fn_coilpos_hdf5 = exp_path + 'sub-'+sub_id+'_matsimnibs.mat'

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Check if files exist
    if not os.path.exists(fn_mesh_roi) or not os.path.exists(fn_coilpos_hdf5):
        print(f"Files missing for subject {sub_id}, skipping.")
        

    try:
        # read subject mesh and roi information
        # ======================================================================

        mesh = mesh_io.read_msh(fn_mesh_roi)
        
        # read coil positions
        # ======================================================================
        pos_matrices = []
        with h5py.File(fn_coilpos_hdf5, 'r') as f:
            ref = f['/matsimnibs'][0,0]
            obj = f[ref][0]
            for r in obj:
                pos_matrices.append(np.array(f[r]).T)
        
        pos_matrices = np.stack(pos_matrices)
        
        start = time.time()
        # Simply call the original function with to_hdf5=False to avoid the tmp_path bug
        dadts = calc_dAdt(
            mesh=mesh, 
            matsimnibs=pos_matrices, 
            coil_path=fn_coil, 
            n_cpus=-1, 
            to_hdf5=False  # Avoid tmp_path bug
        )
        
        # Manually save the results
        np.save(os.path.join(output_path, f'sub-{sub_id}_roi_dadts.npy'), dadts)
        
        processing_time = time.time() - start
        print(f"Subject {sub_id} processed in {processing_time:.2f} seconds")
        print(f"Results saved to {output_path}/sub-{sub_id}_roi_dadts.npy")

    except Exception as e:
        print(f"Error processing subject {sub_id}: {e}")

    print("\nAll subjects processed.")