#%%
from scipy.fft import fft, fftfreq
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
nwbfile.trials.to_dataframe()

# Identify bad channels
electrodes   = nwbfile.electrodes.to_dataframe()
bad_indicies = np.where(electrodes['bad'] == True)

lfp = nwbfile.acquisition['ElectricalSeries'].data[:,:].T

# make a time vector based on the Fs of 3052
Fs   = nwbfile.acquisition['ElectricalSeries'].rate


lfp = nwbfile.acquisition['ElectricalSeries'].data[:30_000,:].T


#%%
# Number of sample points
N = lfp.shape[1]

# sample spacing
T = 1.0 / Fs


yf = fft(lfp[0,:])

xf = fftfreq(N, T)[:N//2]
plt.plot(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
plt.xlim([0,100])
plt.show()
# %%
