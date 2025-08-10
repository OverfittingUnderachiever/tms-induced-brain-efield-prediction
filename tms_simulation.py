import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from simnibs import mesh_io, run_simnibs, sim_struct
from simnibs.simulation import coil_numpy as coil_lib
from joblib import Parallel, delayed
import shutil
import h5py
import fmm3dpy
from tqdm import tqdm

import Code.utils as u

default_simnibs_params = {
    'anisotropy_type': 'vn',
    'aniso_maxratio': 10,
    'aniso_maxcond': 5,
    'didt': 1e6 * 1.49,
    'map_to_vol': False
}


def load_mesh_and_roi(sub_path):
    sub = sub_path.split('/')[-1]
    msh_name = sub + ".msh"
    msh = mesh_io.read_msh(os.path.join(sub_path, 'headmodel', msh_name))

    roi_center_path = os.path.join(sub_path, 'experiment')
    roi_center_name = sub + "_roi_center.mat"
    roi_center = u.loadmat(os.path.join(roi_center_path, roi_center_name))['roi_center']
    
    return msh, roi_center


def rotate_grid(gridz, centernormal_skin):
    theta = np.arccos(np.dot(gridz, centernormal_skin))
    axis_rot = np.cross(gridz, centernormal_skin)
    axis_rot /= np.linalg.norm(axis_rot)
    
    rot_matrix = R.from_rotvec(theta * axis_rot).as_matrix()
    
    return rot_matrix


def generate_grid(centerpoint_skin, gridx_new, gridy_new, search_radius, spatial_resolution):
    n = np.ceil(search_radius/spatial_resolution)
    x_vals = np.arange(-n*spatial_resolution, n*spatial_resolution + spatial_resolution, spatial_resolution)
    y_vals = np.arange(-n*spatial_resolution, n*spatial_resolution + spatial_resolution, spatial_resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    points = centerpoint_skin + X.reshape(-1,1)*gridx_new + Y.reshape(-1,1)*gridy_new
    grid = np.stack((X.flatten(),Y.flatten())).T
    keep = np.sqrt(X.flatten()**2+Y.flatten()**2) <= search_radius

    return points[keep], grid[keep]


def calc_matsimnibs(mesh, grid_centers, distance, rot_angles, headz):
    
    skin_surface = [5, 1005]
    msh_surf = mesh.crop_mesh(elm_type=2)
    msh_skin = msh_surf.crop_mesh(skin_surface)

    z_vectors = np.zeros((len(grid_centers), 3))

    centers, closest = msh_skin.find_closest_element(grid_centers, return_index=True)

    z_vectors = - msh_skin.triangle_normals()[closest]

    centers = centers - distance * z_vectors

    matsimnibs = np.zeros((len(rot_angles), len(grid_centers), 4, 4), dtype=float)
    matsimnibs[:, :, 3, 3] = 1

    for a, rot_angle_deg in enumerate(rot_angles):
        angle_rad = np.deg2rad(rot_angle_deg)
        rotation_vectors = angle_rad * z_vectors
        rot_matrix = R.from_rotvec(rotation_vectors).as_matrix().transpose(0,2,1)

        # Project headz onto the orthogonal complement of z_vectors
        dot_product = np.einsum('ij,j->i', z_vectors, headz)
        norm_squared = np.einsum('ij,ij->i', z_vectors, z_vectors)
        projection = (dot_product / norm_squared)[:, np.newaxis] * z_vectors
        headz_projected = headz - projection

        # Rotate headz_projected using the rotation matrices
        y_vectors = np.einsum('ijk,ik->ij', rot_matrix, headz_projected)
        y_vectors /= np.linalg.norm(y_vectors, axis=1)[:, None]

        # Determine x
        x_vectors = np.cross(y_vectors, z_vectors)
        
        # matsimnibs
        matsimnibs[a, :, :3, 0] = x_vectors
        matsimnibs[a, :, :3, 1] = y_vectors
        matsimnibs[a, :, :3, 2] = z_vectors
        matsimnibs[a, :, :3, 3] = centers
        
    return matsimnibs


# Function to compute initial cylindrical ROI
def compute_cylindrical_roi(mesh, roi_center_gm, skin_normal_avg, roi_radius):
    top = roi_center_gm + (skin_normal_avg * 10)
    base = roi_center_gm - (skin_normal_avg * 30)

    e = base - top
    m = np.cross(top, base)

    # Initialize the mask array
    nodes = mesh.nodes[:]

    # Compute distances and projections
    cross_e_rP = np.cross(e, nodes)
    d = np.linalg.norm(m + cross_e_rP, axis=1) / np.linalg.norm(e)

    # Compute rQ
    rQ = nodes + np.cross(e, m + cross_e_rP) / np.linalg.norm(e)**2

    # Compute weights wA and wB
    wA = np.linalg.norm(np.cross(rQ, base), axis=1) / np.linalg.norm(m)
    wB = np.linalg.norm(np.cross(rQ, top), axis=1) / np.linalg.norm(m)

    cylindrical_roi = (d <= roi_radius) & (wA >= 0) & (wA <= 1) & (wB >= 0) & (wB <= 1)
    return cylindrical_roi


# Function to remove islands of nodes using geodesic distance
def remove_islands(cropped, roi_center):
    _,center_id = cropped.find_closest_element(roi_center.gm, return_index=True)
    comps = cropped.elm.connected_components()
    valid_elms = [c for c in comps if np.isin(center_id, c)][0]
    return cropped.crop_mesh(elements=valid_elms)


def crop_mesh_nodes(mesh, nodes_bool):
    node_keep_indexes = np.append(np.where(nodes_bool)[0] + 1, -1)
    elements_to_keep = np.where(np.all(np.isin(mesh.elm.node_number_list, node_keep_indexes).reshape(-1, 4),axis=1))[0]

    return mesh.crop_mesh(elements=elements_to_keep+1)


def get_matsimnibs(matsim_params, msh, roi_center, gridx=np.array([1, 0, 0]), gridz=np.array([0, 0, 1]), headz=np.array([0, 0, 1])):
    search_radius = matsim_params['search_radius']
    spatial_resolution = matsim_params['spatial_resolution']
    rotation_angles = matsim_params['rotation_angles']
    if np.isscalar(rotation_angles):
        rotation_angles = np.array([rotation_angles])
    distance = matsim_params['distance']

    rot_matrix = rotate_grid(gridz, roi_center.skin_vec)
    gridx_new = rot_matrix @ gridx
    gridy_new = np.cross(roi_center.skin_vec, gridx_new)
    points, s_grid = generate_grid(roi_center.skin, gridx_new, gridy_new, search_radius, spatial_resolution)
    matsimnibs = calc_matsimnibs(msh, points, distance=distance, rot_angles=rotation_angles, headz=headz).reshape(-1,4,4)
    grid = np.stack([[np.array([a, *s]) for s in s_grid] for a in rotation_angles]).reshape(-1,3)
    return matsimnibs, grid


def calc_dAdt(mesh, matsimnibs, coil_path, n_cpus=1, to_hdf5=True, save_path=None, return_data=True):
    print(f"Preparing simulations for {matsimnibs.shape[0]} coil positions...")

    didt = 1000000 * 1.49

    if to_hdf5:
        if save_path is None:
            save_path = os.path.split(coil_path)[0]
            tmp_path = os.path.join(save_path, 'tmp')
            os.mkdir(tmp_path)

    def get_dAdt(matsim, i):
        dAdt = coil_lib.set_up_tms(mesh, coil_path, matsim, didt)[:,:]

        if to_hdf5:
            np.save(tmp_path + '/%d.npy' % i, dAdt)
            return
        else:
            return dAdt

    if n_cpus==1:
        # Run sequentially
        res = []
        for i, matsim in enumerate(matsimnibs):
            res.append(coil_lib.set_up_tms(mesh, coil_path, matsim, didt)[:])
    else:
        res = Parallel(n_jobs=n_cpus)(delayed(get_dAdt)(matsim, i) for i, matsim in enumerate(matsimnibs))
    
    if to_hdf5:
        with h5py.File(save_path + '/dAdts.h5', 'w') as f:
            dAdts = f.create_dataset('dAdt', shape=(len(matsimnibs), len(mesh.elm.triangles), 3))
            for i in range(len(matsimnibs)):
                dAdts[i] = np.load(tmp_path + '/%d.npy' % i)
        os.system("rm -r %s" % tmp_path)
        if return_data:
            dAdts = u.h5py_dataset(save_path + '/dAdts.h5', 'dAdt')[:]
    else:
        dAdts = np.stack(res)
        if not save_path is None:
            np.save(save_path + '/dAdts.npy')
    if return_data:
        return dAdts
    

def get_dAdt_from_coord(coil_matrices, pos, coil_file, didt=1e6*1.49):
    A = np.zeros((len(coil_matrices), len(pos), 3))

    # Read coil data
    d_position, d_moment = coil_lib.read_ccd(coil_file)
    d_position = np.hstack([d_position * 1e3, np.ones((d_position.shape[0], 1))])

    # Precompute transformed positions
    d_pos_all = np.array([coil_matrix.dot(d_position.T).T[:, :3] * 1e-3 for coil_matrix in coil_matrices])

    # Precompute rotated moments
    d_mom_all = np.array([np.dot(d_moment, coil_matrix[:3, :3].T) for coil_matrix in coil_matrices])

    for i in range(len(coil_matrices)):
        d_pos = d_pos_all[i]
        d_mom = d_mom_all[i]

        out = fmm3dpy.lfmm3d(
            eps=1e-3,
            sources=d_pos.T,
            charges=d_mom.T,
            targets=pos.T*1e-3,
            pgt=2,
            nd=d_mom.shape[-1]
            )

        # Compute the components of A
        A[i, :, 0] = (out.gradtarg[1,2] - out.gradtarg[2,1])
        A[i, :, 1] = (out.gradtarg[2,0] - out.gradtarg[0,2])
        A[i, :, 2] = (out.gradtarg[0,1] - out.gradtarg[1,0])

    A *= -1e-7 * didt

    return A

def get_skin_average_normal_vector(mesh, roi_center, roi_radius):
    # Extract skin region
    skin_region_id = 1005  # Assuming region_idx 1005 corresponds to skin
    skin_cells = mesh.crop_mesh(tags=skin_region_id)
    
    # Get skin triangle centers and normals
    skin_centers = skin_cells.elements_baricenters()[:]
    skin_normals = skin_cells.triangle_normals()[:]
    
    # Compute skin ROI
    roi_center_skin = roi_center.skin
    distances = np.linalg.norm(skin_centers - roi_center_skin, axis=1)
    skin_roi = distances < roi_radius
    
    # Average normal vector in the ROI
    skin_normal_avg = np.mean(skin_normals[skin_roi], axis=0)
    
    return skin_normal_avg


def extract_save_efield(sim_path, roi_center, skin_normal_avg, clean_path=False, save_final_mesh=False, save_path_mesh=None):
    # Load mesh
    msh = mesh_io.read_msh(sim_path)
    
    # Get roi_center_gm attribute or key
    if hasattr(roi_center, 'gm'):
        roi_center_gm = roi_center.gm
    else:
        roi_center_gm = roi_center['gm']
            
    # Compute cylindrical ROI
    cylindrical_roi = compute_cylindrical_roi(msh, roi_center_gm, skin_normal_avg, roi_radius=20)
    
    # Crop mesh using cylindrical ROI
    cropped = crop_mesh_nodes(msh, cylindrical_roi)
    
    # Remove islands from cropped mesh
    final_roi = remove_islands(cropped, roi_center)
    
    # Save final mesh if requested
    if save_final_mesh and save_path_mesh is not None:
        final_roi.write(save_path_mesh)
    
    # Extract E-field
    efield = final_roi.nodedata[0][:]
    
    # Clean up if requested
    if clean_path:
        parent_path = os.path.abspath(os.path.join(os.path.dirname(sim_path), ".."))
        try:
            shutil.rmtree(parent_path)
        except Exception:
            pass
    
    return efield


def run_efield_sim(simnibs_params, matsimnibs, n_cpus=1):
    # Create SimNIBS session
    s = sim_struct.SESSION()
    s.map_to_surf = simnibs_params['map_to_surf']
    s.fields = simnibs_params['fields']
    s.fnamehead = simnibs_params['mesh_path']
    s.pathfem = simnibs_params['out_path']
    s.open_in_gmsh = False
    #s.subpath = '/home/freyhe/MA_Henry/data/sub-005/headmodel/m2m_sub-005'

    # Set the subpath directly to the headmodel directory

    
    # Create TMS list
    tms_list = s.add_tmslist()
    tms_list.fnamecoil = simnibs_params['coil_path']
    tms_list.anisotropy_type = simnibs_params['anisotropy_type']
    tms_list.fn_tensor_nifti = simnibs_params['nifti_path']
    tms_list.aniso_maxratio = simnibs_params['aniso_maxratio']
    tms_list.aniso_maxcond = simnibs_params['aniso_maxcond']
    tms_list.name = 'sim'
    
    # Add positions based on matsimnibs shape
    if len(matsimnibs.shape) == 3:
        for m in matsimnibs:
            pos = tms_list.add_position()
            pos.matsimnibs = m
            pos.didt = simnibs_params['didt']
    else:
        pos = tms_list.add_position()
        pos.matsimnibs = matsimnibs
        pos.didt = simnibs_params['didt']
    
    # Run simulation
    run_simnibs(s, cpus=n_cpus)
    
    # Get output path
    coil_name = os.path.splitext(os.path.basename(tms_list.fnamecoil))[0]
    sim_path = s.pathfem + "/subject_overlays/{0}-{1:0=4d}_{2}_".format(tms_list.name, 1, coil_name) + tms_list.anisotropy_type + '_central.msh'
    
    return sim_path


def run_exp(params, simnibs_params, n_cpus=-1, n_batches=None, batch_n=None):
    workspace = params['workspace']
    sub_id = params['sub_id']
    exp_type = params['exp']
    
    # Set paths
    sub_path = workspace + '/data/sub-' + sub_id
    exp_path = sub_path + '/experiment/' + exp_type
    
    fn_coil = workspace + '/data/coil/MagVenture_Cool-B65.ccd'
    fn_tensor_nifti = sub_path + '/headmodel/d2c_sub-' + sub_id + '/dti_results_T1space/DTI_conf_tensor.nii.gz'
    fn_mesh_roi = exp_path + '/sub-' + sub_id + '_middle_gray_matter_roi.msh'
    
    # Load mesh and ROI
    msh, roi_center = load_mesh_and_roi(sub_path)
    fn_mesh = msh.fn
    
    # Assign simnibs session parameters
    simnibs_params['coil_path'] = fn_coil
    simnibs_params['nifti_path'] = fn_tensor_nifti
    if 'mesh_path' not in simnibs_params:
        simnibs_params['mesh_path'] = fn_mesh
    simnibs_params['sim_path'] = os.path.join(exp_path, 'tmp')
    
    # Add default parameters if missing
    for key in default_simnibs_params.keys():
        if key not in simnibs_params.keys():
            simnibs_params[key] = default_simnibs_params[key]
    
    # Create necessary paths
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    
    # Clean and recreate simulation directory
    if os.path.isdir(simnibs_params['sim_path']):
        shutil.rmtree(simnibs_params['sim_path'])
    
    os.makedirs(simnibs_params['sim_path'], exist_ok=True)
    
    # Get normal vector
    skin_normal_avg = get_skin_average_normal_vector(msh, roi_center, roi_radius=20)
    
    # Create matsimnibs file of locations
    matsimnibs, grid = get_matsimnibs(params, msh, roi_center)
    
    # Save data if not batched or first batch
    if (n_batches is None) or (batch_n == 0):
        np.save(exp_path + '/sub-' + sub_id + '_matsimnibs.npy', matsimnibs)
        np.save(exp_path + '/sub-' + sub_id + '_grid.npy', grid)
    
    # Split into batch if needed
    if n_batches:
        matsimnibs = np.array_split(matsimnibs, n_batches)[batch_n]
    
    # Define simulation function
    def sim_get_e(matsim, i):
        out_path = os.path.join(simnibs_params['sim_path'], str(batch_n or '') + '_' + str(i))
        
        # Clean and create output directory
        if os.path.isdir(out_path):
            shutil.rmtree(out_path)
        
        os.makedirs(out_path, exist_ok=True)
        
        simnibs_params['out_path'] = out_path
        
        # Run simulation and extract E-field
        sim_path = run_efield_sim(simnibs_params, matsim)
        e = extract_save_efield(
            sim_path, 
            roi_center, 
            skin_normal_avg, 
            save_final_mesh=True, 
            save_path_mesh=fn_mesh_roi, 
            clean_path=True
        )
        return e
    
    print('-' * 20)
    print('Running %d efield simulations' % len(matsimnibs))
    
    # Run simulations in parallel
    efields = Parallel(n_jobs=n_cpus)(delayed(sim_get_e)(matsim, i) for i, matsim in enumerate(tqdm(matsimnibs)))
    
    # Clean up
    os.system("rm -r %s" % simnibs_params['sim_path'])
    
    # Save results
    if batch_n is None:
        np.save(exp_path + '/sub-' + sub_id + '_efields.npy', np.stack(efields))
    else:
        efield_dir = exp_path + '/efield_sims'
        if not os.path.isdir(efield_dir):
            os.mkdir(efield_dir)
        np.save(efield_dir + f'/sub-{sub_id}_efields_{batch_n}.npy', np.stack(efields))