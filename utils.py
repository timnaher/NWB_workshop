from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from fsspec import filesystem
from h5py import File
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from scipy.signal import decimate
import numpy as np
from scipy.signal import butter, lfilter, hilbert
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from IPython.display import display, clear_output
import time
import scipy.ndimage as ndimage




#___________ io tools ___________
def load_nwbfile(dandiset_id,file_path):
    # add comment
    with DandiAPIClient() as client:
        asset  = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(file_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    # Create a virtual filesystem based on the http protocol and use caching to save accessed data to RAM.
    fs          = filesystem("http")
    file_system = fs.open(s3_url, "rb")
    file        = File(file_system, mode="r")
    io          = NWBHDF5IO(file=file, load_namespaces=True)
    nwbfile     = io.read()
    return nwbfile


#___________ signal processing tools ___________
def zscore(data):
    return (data - np.mean(data)) / np.std(data)



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




#___________ plot tools ___________

def plot_movie(data,name):
    fig_lfp, ax_lfp     = plt.subplots()
    fig_phase, ax_phase = plt.subplots()
    ims_phase,ims_lfp   = [],[]

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    for i in range(data.shape[2]):  # 100 frames
        # smooth the current matrix frame
        frame = data[:,:,i]
        phase = np.angle(hilbert(frame))

        #frame = ndimage.gaussian_filter(frame, sigma=0.5)
        im_phase = ax_phase.imshow(phase,vmin=-np.pi, vmax=np.pi,cmap='twilight')
        ims_phase.append([im_phase])

        im_lfp   = ax_lfp.imshow(frame,vmin=vmin,vmax=vmax,cmap='RdBu')
        ims_lfp.append([im_lfp])


    ani_lfp   = animation.ArtistAnimation(fig_lfp, ims_lfp, interval=50, blit=True,repeat_delay=500)
    ani_phase = animation.ArtistAnimation(fig_phase, ims_phase, interval=50, blit=True,repeat_delay=500)

    writer = PillowWriter(fps=30)

    ani_lfp.save(    f"lfp_{name}.gif", writer=writer)
    ani_phase.save(f"phase_{name}.gif", writer=writer)
