#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from utils import *
from flow_utils import *
from multiprocessing import Pool

# loop over sessions
df_glob      = pd.DataFrame()
sessions     = ['B105','B15','B76','B8','B89','B9','B1']
dandiset_id  = "000019"
sub          = 'EC2'

#%%
for ses in sessions:
    print(ses)
    file_path   = f'sub-{sub}/sub-{sub}_ses-{sub}-{ses}.nwb'
    nwbfile     = load_nwbfile(dandiset_id,file_path)
    electrodes  = nwbfile.electrodes.to_dataframe()
    df_sub      = pd.read_pickle(f'df_{ses}.pkl')
    df_glob     = pd.concat([df_glob, df_sub], ignore_index=True)

df_glob.to_pickle(f'/home/jovyan/NWB_workshop/data/grand_df_{sub}.pkl')



#%% Try to get vector fields
df_glob = pd.read_pickle(f'/home/jovyan/NWB_workshop/data/grand_df_{sub}.pkl')



def process_row(index):
    try:
        # Retrieve the row from the global dataframe using the provided index
        row = df_glob.iloc[index]
        # Preprocess the row data for optical flow analysis
        data = zscore_3d(row.lfp.reshape(16, 16, -1))
        # Calculate the optical flow
        x, y, u, v = opticalFlowHS(data, alpha=1, max_iter=100, wait_bar=False)
        
        # Calculate the mean vector field along the third dimension
        u_mean = np.nanmean(u, axis=2)
        v_mean = np.nanmean(v, axis=2)
        
        # Precompute gradients to avoid redundant calculations
        grad_u_mean = np.gradient(u_mean)
        grad_v_mean = np.gradient(v_mean)
        
        # Compute the divergence as the sum of specific gradients
        div = grad_u_mean[0] + grad_v_mean[1]
        
        # Compute the curl as the difference of specific gradients
        curl = grad_u_mean[1] - grad_v_mean[0]
        
        return u, v, div, curl
    
    except Exception as e:
        # Handle potential errors, possibly by logging or raising a custom error
        print(f"An error occurred: {e}")
        # Depending on your error handling strategy, you might return None or raise an error
        return None


if __name__ == '__main__':
    indicies =  range(len(df_glob))
    indicies = indicies[:10]

    with Pool(96) as pool:  # use all 96 CPUs
        results = pool.map(process_row, indicies )

    # combine the results with df_glob
    for i, result in enumerate(results):
        df_glob  = pd.concat([df_glob , pd.DataFrame([result], columns=['u', 'v', 'div', 'curl'])], ignore_index=True)

    # save the dataframe to disk
    df_glob.to_pickle(f'/home/jovyan/NWB_workshop/data/grand_df_{sub}_with_vector_fields.pkl')

# %%
U = np.mean(results[0][0],axis=2)
V = np.mean(results[0][1],axis=2)

plt.figure()
#quiver
plt.quiver(U,V)
