#%%
import numpy as np
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from fsspec import filesystem
from h5py import File
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from scipy.signal import decimate
from utils import load_nwbfile
from scipy.signal.windows import blackman


dandiset_id = "000019"
file_path   = 'sub-EC2/sub-EC2_ses-EC2-B89.nwb'
nwbfile     = load_nwbfile(dandiset_id,file_path)
#nwbfile.trials.to_dataframe()

#%%
# Identify bad channels
electrodes   = nwbfile.electrodes.to_dataframe()
bad_indicies = np.where(electrodes['bad'] == True)

lfp = nwbfile.acquisition['ElectricalSeries'].data[:,:].T

# make a time vector based on the Fs of 3052
Fs   = nwbfile.acquisition['ElectricalSeries'].rate


lfp = nwbfile.acquisition['ElectricalSeries'].data[:10_000,:].T

# take every third element of lfp
lfp = lfp[:,::3]


#%%
# bandpass filter the data in the beta range
low_bound, up_bound = 12, 20
lfp_filt = bandpass_filter_fast(lfp,low_bound,up_bound,Fs)

#%% run a pca on the data and project in 3d. use scipy
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(lfp_filt.T)


# project the data into 3d
lfp_pca = pca.transform(lfp_filt.T)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lfp_pca[:,0],lfp_pca[:,1],lfp_pca[:,2])
plt.show()

