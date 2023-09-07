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
from scipy.signal import butter, filtfilt

def bandpass_filter_fast(data, lowcut, highcut, fs, order=3):
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

Fs      = 3051.7578
band = 'beta'
if band == 'beta':
    low_bound, up_bound = 12, 20
elif band == 'alpha':
    low_bound, up_bound = 7, 11


sessions    = ['B105','B15','B76','B8','B89','B9','B1']
dandiset_id = "000019"
sub         = 'EC2'

for ses in sessions:
    print(ses)
    file_path   = f'sub-EC2/sub-EC2_ses-EC2-{ses}.nwb'
    nwbfile     = load_nwbfile(dandiset_id,file_path)
    nwbfile.trials.to_dataframe()

    # Identify bad channels
    electrodes   = nwbfile.electrodes.to_dataframe()
    bad_indicies = np.where(electrodes['bad'] == True)
    lfp          = nwbfile.acquisition['ElectricalSeries'].data[:].T

    Fs   = nwbfile.acquisition['ElectricalSeries'].rate
    T    = (lfp.shape[1]/Fs)+20 # Duration in seconds
    time = np.arange(0, T, 1/Fs)

    # truncate time to the length of lfp ont he first dim
    time = time[:lfp.shape[1]]

    # get the trial strucutre
    trials = nwbfile.trials.to_dataframe()


    # filter the data here:
    print('filtering')
    lfp_filt = bandpass_filter_fast(lfp,low_bound,up_bound,Fs)
    print('done filtering')

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
        trial = lfp_filt[:,start_idx:end_idx]

        # make dict
        mydict = {'lfp':trial, 'transition_time':transition_idx,'condition':trials.condition.iloc[itrial]}

        # append to df
        df = df.append(mydict,ignore_index=True)


    # make new conditions based on last 2 letters of word spoken
    df['vowel'] = df.condition.str[-1:]
    df          = df.assign(id=(df['vowel']).astype('category').cat.codes)


    # save the df to disk
    print('saving')
    df.to_pickle(f'df_{ses}_{band}.pkl')
    print('done saving')
# %%
