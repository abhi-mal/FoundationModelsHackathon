import h5py
import numpy as np
import os
import glob

def create_ad_dataset(real_glob, gen_glob, output_path):
    # 1. Grab ALL Real SR data batches
    real_files = glob.glob(real_glob)
    # 2. Grab ALL Fake Generated data batches
    fake_files = glob.glob(gen_glob)
    
    if not real_files or not fake_files:
        raise ValueError(f"Missing Real or Fake files.")
    
    print(f"Found {len(real_files)} Real files and {len(fake_files)} Fake files.")
    
    # 3. Load all Fake data 
    X_fake_list = []
    for f in fake_files:
        with h5py.File(f, 'r') as h5f:
            X_fake_list.append(h5f['data'][:])
    X_fake = np.concatenate(X_fake_list, axis=0)
    
    # 4. Load just enough Real data to symmetrically balance the Fake data
    X_real_list = []
    real_loaded = 0
    for f in real_files:
        with h5py.File(f, 'r') as h5f:
            remaining = len(X_fake) - real_loaded
            if remaining <= 0:
                break
            batch_data = h5f['data'][:remaining] 
            X_real_list.append(batch_data)
            real_loaded += len(batch_data)
            
    X_real = np.concatenate(X_real_list, axis=0)
    
    print(f"Loaded {len(X_fake)} Fake events and {len(X_real)} Real events.")

    # 5. Combine data and Assign Binary Labels (0 = Fake, 1 = Real)
    X = np.concatenate([X_fake, X_real], axis=0)
    y = np.concatenate([np.zeros(len(X_fake)), np.ones(len(X_real))], axis=0)
    
    # 6. Shuffle the dataset so real and fake are mixed
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    # 7. Write out the Custom dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('data', data=X)
        f_out.create_dataset('pid', data=y)

# Example usage
create_ad_dataset('omni_data/aspen_top_ad_sr/test/omni_RunG_*.h5', 
                  'omni_data/aspen_top_ad_sr/test/generated_fine_tune*.h5', 
                  'omni_data/custom/train/train.h5')