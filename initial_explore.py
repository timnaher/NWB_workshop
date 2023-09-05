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

sub         = 'Roqui'
dandiset_id = "000410"
file_path   = f"sub-{sub}/sub-{sub}_ses-{sub}-01_behavior+ecephys.nwb" 
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
# get the timestaps:
times   = nwbfile.epochs.to_dataframe()
series1 = times.iloc[2].stop_time - times.iloc[2].start_time
print(series1)

# notes
#%%
f = h5py.File(file_system, 'r')
#print(f.keys())
#print(f['intervals']['epochs'].keys())
print(f['processing'].keys())

#%%



start_times         = np.array(f['intervals']['epochs']['start_time'])
end_times           = np.array(f['intervals']['epochs']['stop_time'])
session_description = f['session_description']

lfp            = f['acquisition']['e-series']['data']
lfp_timestamps = f['acquisition']['e-series']['timestamps']


# Decimate data to 1kHz
temp = lfp[:300000,0]
temp = decimate(temp, 29, axis=0) 

# Run FFT and get power spectrum
fft_vals = np.fft.fft(temp)
power_spectrum = np.abs(fft_vals[:len(temp)//2])**2 / len(temp) * 2

# Get frequency bins for plotting
freq_bins = np.fft.fftfreq(len(temp), 1/1000.)[:len(temp)//2]

# Select frequency range, e.g., 0-20Hz (or whichever range you desire)
mask = (freq_bins >= 0) & (freq_bins <= 100)

# Plot power spectrum within desired frequency range
plt.plot(freq_bins[mask], power_spectrum[mask])
plt.xlim(0, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.show()


#%%

stamps = np.array(lfp_timestamps[60_000_000:80_000_000])
plt.plot(np.abs(stamps - start_times[2]))


#%%



print(lfp.shape)

#for beg,end in zip(start_times,end_times):
#    # find beg indx in lfp_timestamps
#    beg_idx = np.where(lfp_timestamps == beg)[0][0]
#    end_idx = np.where(lfp_timestamps == end)[0][-1]


#sample_count = f['processing']['sample_count']['sample_count']['data']
#position = f['processing']['behavior']['position']['series_3']['data']

#plt.plot(sample_count[:100000])
#plt.plot(sample_count[:])
#analog_data = f['processing']['analog']['analog']['analog']['data']


# notes
#f['specifications'] is not important
#f['session_start_time] is empty
#  f['processing']['analog']['analog']['analog']['data'] is empty/just zero
# sample count not sure
# f['session_description'] is empty



#%%
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.plot(subset[:20000,1:5])
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Voltage (uV)')
ax.set_title('Raw LFP Data')
#%%
# get the keys of the hdf5 file
list(f.keys())

# get the data
data = f['acquisition']['e-series']['data']

with h5py.File('path_to_your_file.h5', 'r') as f:
    subset = f['data'][:10000, :]

#%%
# confert do pandas dataframe
import pandas as pd

df = pd.DataFrame(data=data)
df.head()

# %%
