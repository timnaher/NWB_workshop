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

dandiset_id = "000019"
file_path   = 'sub-EC2/sub-EC2_ses-EC2-B1.nwb'
# Get the location of the file on DANDI
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(file_path)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

# Create a virtual filesystem based on the http protocol and use caching to save accessed data to RAM.
fs = filesystem("http")
file_system = fs.open(s3_url, "rb")
file = File(file_system, mode="r")
# Open the file with NWBHDF5IO
io = NWBHDF5IO(file=file, load_namespaces=True)

nwbfile = io.read()


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


def zscore(data):
    return (data - np.mean(data)) / np.std(data)


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

electrodes = nwbfile.electrodes.to_dataframe()

# plot the x and y coordinates of the electrodes as scatter
plt.scatter(electrodes.x, electrodes.y, s=1)
plt.scatter(electrodes.x[0], electrodes.y[0], s=10)


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
