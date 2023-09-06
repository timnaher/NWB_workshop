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
from scipy.signal import butter, lfilter




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