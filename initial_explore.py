#%%
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from fsspec import filesystem
from h5py import File
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt

dandiset_id = "000410"
file_path   = "sub-Jaq/sub-Jaq_ses-jaq-01_behavior+ecephys.nwb" # file size ~67GB
#sub-Jaq/sub-Jaq_ses-Jaq-01_behavior+ecephys.nwb

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
# %%
import h5py
data = nwbfile.acquisition['e-series'].data

# load the hdf5 file
f = h5py.File(file_system, 'r')

with h5py.File(file_system, 'r') as f:
    subset = f['acquisition']['e-series']['data'][:10000, :]


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
