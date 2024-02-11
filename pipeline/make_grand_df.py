#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from IPython.display import display, clear_output
import time
import scipy.ndimage as ndimage
from utils import *

# loop over sessions
df_glob      = pd.DataFrame()
sessions    = ['B105','B15','B76','B8','B89','B9','B1']
dandiset_id = "000019"
sub        = 'EC2'

for ses in sessions:
    print(ses)
    file_path   = f'sub-{sub}/sub-{sub}_ses-{sub}-{ses}.nwb'
    nwbfile     = load_nwbfile(dandiset_id,file_path)
    electrodes  = nwbfile.electrodes.to_dataframe()


    df_sub      = pd.read_pickle(f'df_{ses}.pkl')
    df_glob     = pd.concat([df_glob, df_sub], ignore_index=True)

df_glob.to_pickle(f'/home/jovyan/NWB_workshop/data/grand_df_{sub}.pkl')



#%% Try to get vector fields

df = pd.DataFrame()
from tqdm import tqdm
from utils import *
from flow_utils import *

df['u'] = None
df['v'] = None
for j in tqdm(range(len(df_glob))):
    data = zscore_3d(df.iloc[j].lfp.reshape(16,16,-1))
    x, y, u, v = opticalFlowHS(data, alpha=1, max_iter=100, wait_bar=False)

    # save u and v in the dataframe
    df.loc[j, 'u'] = [u]
    df.loc[j, 'v'] = [v]


df['u_pre']     = None
df['v_pre']     = None
df['u_post']    = None
df['v_post']    = None
df['div_pre']   = None
df['div_post']  = None
df['curl_pre']  = None
df['curl_post'] = None

for row in tqdm(df.iterrows()):
    row = row[1]

    # compute the pre and post transition vector fields
    u_pre  = np.nanmean(row.u[0][:,:,:row.transition_time],axis=2)
    v_pre  = np.nanmean(row.v[0][:,:,:row.transition_time],axis=2)
    u_post = np.nanmean(row.u[0][:,:,row.transition_time:],axis=2)
    v_post = np.nanmean(row.v[0][:,:,row.transition_time:],axis=2)

    # compute the divergence of the vector fields
    div_pre  = np.gradient(u_pre)[0]  + np.gradient(v_pre)[1]
    div_post = np.gradient(u_post)[0] + np.gradient(v_post)[1]

    # compute the curl
    curl_pre  = np.gradient(u_pre)[1]  - np.gradient(v_pre)[0]
    curl_post = np.gradient(u_post)[1] - np.gradient(v_post)[0]

    # append to the dataframe
    df.loc[row.name,'u_pre']     = [u_pre]
    df.loc[row.name,'v_pre']     = [v_pre]
    df.loc[row.name,'u_post']    = [u_post]
    df.loc[row.name,'v_post']    = [v_post]

    df.loc[row.name,'div_pre']   = [div_pre]
    df.loc[row.name,'div_post']  = [div_post]

    df.loc[row.name,'curl_pre']  = [curl_pre]
    df.loc[row.name,'curl_post'] = [curl_post]















# %%
# import logisitic regression
from sklearn.linear_model import LogisticRegression

clf3 = Pipeline([
    ('Cov', XdawnCovariances()),
    ('TS', TSclassifier()) 
])


clf3_param_grid = {
    'Cov__estimator': ['oas'],
    'Cov__nfilter': [2,4],
    'TS__clf': [LogisticRegression(C=1), LogisticRegression(C=10), LogisticRegression(C=100), LogisticRegression(C=1000)]
}


for row in df_glob.iterrows():
    row = row[1]
    # compute the covariance matrix