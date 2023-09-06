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
from utils import *

dandiset_id = "000019"
file_path   = 'sub-EC2/sub-EC2_ses-EC2-B89.nwb'
nwbfile     = load_nwbfile(dandiset_id,file_path)

nwbfile.trials.to_dataframe()

#%%
# Identify bad channels
electrodes   = nwbfile.electrodes.to_dataframe()
bad_indicies = np.where(electrodes['bad'] == True)

lfp = nwbfile.acquisition['ElectricalSeries'].data[:,:].T

# make a time vector based on the Fs of 3052


Fs   = nwbfile.acquisition['ElectricalSeries'].rate
T    = (lfp.shape[1]/Fs)+20 # Duration in seconds
time = np.arange(0, T, 1/Fs)

# truncate time to the length of lfp ont he first dim
time = time[:lfp.shape[1]]


# get the trial strucutre
trials = nwbfile.trials.to_dataframe()
df     = pd.DataFrame(columns=['lfp','transition_time','condition'])

for itrial in np.arange(trials.shape[0]):
    best_fit_start      = np.abs(time - trials.start_time.iloc[itrial])
    best_fit_end        = np.abs(time - trials.stop_time.iloc[itrial])
    best_fit_transition = np.abs(time - trials.cv_transition_time.iloc[itrial])
    # find the index of start and stop
    start_idx = np.argmin(best_fit_start)
    end_idx   = np.argmin(best_fit_end)
    transition_idx = np.argmin(best_fit_transition) - start_idx # relative to onset

    # epoch the data
    trial = lfp[:,start_idx:end_idx]

    # make dict
    mydict = {'lfp':trial, 'transition_time':transition_idx,'condition':trials.condition.iloc[itrial]}

    # append to df
    df = df.append(mydict,ignore_index=True)


# save the df to disk
df.to_pickle()

#%% compute the ERP for the condition raa



df_subcond = df[df.condition=='shee']
# get all the lengths
lengths = []
for i,row in enumerate(df_subcond.iterrows()):
    row = row[1]
    lengths.append(row.lfp.shape[1])

# find minimum length
maxlength = np.max(lengths)

for i,row in enumerate(df_subcond.iterrows()):
    row = row[1]
    thistrial = row.lfp
    # subtract the mean
    thistrial = thistrial -  np.nanmean(thistrial,axis=1)[:,None]
    if thistrial.shape[1] < maxlength:
        # pad with zeros
        pads      = np.ones((256,maxlength - thistrial.shape[1])) * np.nan
        thistrial = np.column_stack((thistrial,pads))
    if i == 0:
        erp = thistrial
    else:
        erp = np.nansum(np.dstack((erp,thistrial)),2)


grand_erp = erp/i

plt.plot(grand_erp[:10,:].T)
#%%
# make bad channels nan
lfp[bad_indicies,:] = np.nan

# make grid data
lfp = lfp.reshape(16,16,-1)



#%%





#%%

# reorder in 16 x 16 grid
lfp = lfp.reshape(16,16,-1)

# make electrode 9 and 12 nan
#lfp[12,9,:] = np.nan

# interpolate the electrode based on the surrounding electrodes
lfp[12,9,:] = lfp[12,10,:]



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

lfp = zscore_3d(lfp)
lfp = decimate(lfp, 3, axis=2)
lfp = bandpass_filter_3d(lfp, 15, 25, Fs)


#%%
electrodes = nwbfile.electrodes.to_dataframe()

# plot the x and y coordinates of the electrodes as scatter
plt.scatter(electrodes.y, electrodes.z, s=1)
plt.scatter(electrodes.y[16], electrodes.z[16], s=10)



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


fig_lfp, ax_lfp     = plt.subplots()
fig_phase, ax_phase = plt.subplots()

ims_phase = []
ims_lfp   = []



data = df.iloc[0].lfp

data = data.reshape(16,16,-1)
data = zscore_3d(data)
data = bandpass_filter_3d(lfp, 15, 25, Fs)
vmin = np.nanmin(data)
vmax = np.nanmax(data)

for i in range(data.shape[2]):  # 100 frames
    # smooth the current matrix frame
    frame = data[:,:,i]
    phase = np.angle(hilbert(frame))

    #frame = ndimage.gaussian_filter(frame, sigma=0.5)
    im_phase = ax_phase.imshow(phase,vmin=-np.pi, vmax=np.pi,cmap='twilight')
    ims_phase.append([im_phase])

    im_lfp   = ax_lfp.imshow(frame,vmin=vmin,vmax=vmax,cmap='viridis')
    ims_lfp.append([im_lfp])


ani_lfp   = animation.ArtistAnimation(fig_lfp, ims_lfp, interval=50, blit=True,repeat_delay=500)
ani_phase = animation.ArtistAnimation(fig_phase, ims_phase, interval=50, blit=True,repeat_delay=500)

writer = PillowWriter(fps=50)

ani_lfp.save("lfp_demo_8-12.gif", writer=writer)
ani_phase.save("phase_demo_8-12.gif", writer=writer)


    
    
# %%
from scipy.signal import hilbert



data = df.iloc[0].lfp
# find the min and max of all frames for coloraxis

data = data.reshape(16,16,-1)
data = zscore_3d(data)
data = bandpass_filter_3d(lfp, 16, 23, Fs)

# get the phase of the data via hilbert
data_a = 
phase  = np.angle(hilbert(data))
# %%
