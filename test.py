from scipy import io
import numpy as np
# Load the .mat file
data = io.loadmat('/home/ninad/vaibhav_r/GNR_650/Assignment_2/IAB-GZSL/data/AWA2/vitL14.mat')

# Print all the keys in the .mat file
print("Keys in the .mat file:", data.keys())

# Loop through each key to inspect its content
for key in data:
    if not key.startswith('__'):  # Skip metadata keys that start with '__'
        print(f"\nKey: {key}")
        print(f"Type: {type(data[key])}")
        
        # If it's an array, print shape and data type
        if isinstance(data[key], np.ndarray):
            print(f"Shape: {data[key].shape}")
            print(f"Dtype: {data[key].dtype}")
            
            # Optionally print a small sample of the data
            print("Sample Data:", data[key][:5] if data[key].size > 5 else data[key])
