import numpy as np

file_path = "/home/freyhe/MA_Henry/data/sub-000/experiment/simulation_results/new_simulation/dadt_sims/000_dadt_1.npy"

# Load the .npy file
data = np.load(file_path, allow_pickle=True)

# Print basic information
print("Data Type:", type(data))
print("Shape:", data.shape)
print("First few elements:", data[:5])  # Adjust based on the data size
