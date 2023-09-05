#%%
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

dandiset_id = "000019"
file_path   = 'sub-EC2/sub-EC2_ses-EC2-B89.nwb'
nwbfile     = load_nwbfile(dandiset_id,file_path)

nwbfile.epochs.to_dataframe()

#%%
# Identify bad channels
electrodes   = nwbfile.electrodes.to_dataframe()
bad_indicies = np.where(electrodes['bad'] == True)

lfp = nwbfile.acquisition['ElectricalSeries'].data[:30_000,:].T

# get the trial strucutre
trials = nwbfile.trials.to_dataframe()

# the trial begings at start_time and ends at stop_time in seconds


# get the sam

# find chanel outlier based on zscore
def zscore(data):
    return (data - np.mean(data)) / np.std(data)


zscore_data  = zscore(lfp)
zscore_data  = np.nanmean(zscore_data, axis=1)
zscore_data  = np.abs(zscore_data)
bad_indicies = np.where(zscore_data > 3)[0]

# make bad channels nan
lfp[bad_indicies,:] = np.nan


# make grid data
lfp = lfp.reshape(16,16,-1)











#%%
lfp = nwbfile.acquisition['ElectricalSeries'].data[:30000,:].T

# reorder in 16 x 16 grid
lfp = lfp.reshape(16,16,-1)

# make electrode 9 and 12 nan
#lfp[12,9,:] = np.nan

# interpolate the electrode based on the surrounding electrodes
lfp[12,9,:] = lfp[12,10,:]

plt.imshow(lfp[:,:,1])


import numpy as np
from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def bandpass_filter_3d(array_3d, lowcut, highcut, fs, order=2):
    filtered_data = np.empty_like(array_3d)
    for i in range(array_3d.shape[0]):
        for j in range(array_3d.shape[1]):
            filtered_data[i, j] = bandpass_filter(array_3d[i, j], lowcut, highcut, fs, order)
    
    return filtered_data



def zscore_3d(array_3d):
    zscore_data = np.empty_like(array_3d)
    for i in range(array_3d.shape[0]):
        for j in range(array_3d.shape[1]):
            zscore_data[i, j] = zscore(array_3d[i, j])
    
    return zscore_data

# Assuming data_3d is your 3D numpy array with shape (16, 16, time)
# fs is your sampling frequency

fs  = 1000  # Example: Sampling frequency is 100 Hz
lfp = zscore_3d(lfp)
lfp = decimate(lfp, 3, axis=2)
lfp = bandpass_filter_3d(lfp, 8, 12, fs)

plt.plot(lfp[0,0,:])

#%%
electrodes = nwbfile.electrodes.to_dataframe()

# plot the x and y coordinates of the electrodes as scatter
plt.scatter(electrodes.y, electrodes.z, s=1)
plt.scatter(electrodes.y[16], electrodes.z[16], s=10)

#%%

# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from IPython.display import display, clear_output
import time
import scipy.ndimage as ndimage


fig, ax = plt.subplots()
ims = []
# set colormap to bte RdBu
plt.set_cmap('RdBu')

time2plot = np.arange(2200,2800)
# find the min and max of all frames for coloraxis
vmin = np.nanmin(lfp[:,:,time2plot])
vmax = np.nanmax(lfp[:,:,time2plot])

for i in np.arange(2200,2800):  # 100 frames
    # smooth the current matrix frame
    frame = lfp[:,:,i]
    frame = ndimage.gaussian_filter(frame, sigma=0.5)

    im = plt.imshow(frame)
    
    # set the coloraxis to be the same for all plots
    plt.clim(vmin, vmax)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=500)

writer = PillowWriter(fps=30)
ani.save("demo_8-12.gif", writer=writer)

    
    
# %%
