#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from IPython.display import display, clear_output
import time
import scipy.ndimage as ndimage
from utils import *

# loop over sessions
df_glob      = pd.DataFrame()
sessions    = ['B105','B15','B76','B8','B89','B9','B1']
dandiset_id = "000019"

for ses in sessions:
    file_path   = f'sub-EC2/sub-EC2_ses-EC2-{ses}.nwb'
    nwbfile     = load_nwbfile(dandiset_id,file_path)
    electrodes  = nwbfile.electrodes.to_dataframe()
    df_sub      = pd.read_pickle(f'df_{ses}vecfields_beta.pkl')
    df_glob     = pd.concat([df_glob, df_sub], ignore_index=True)

df_glob.to_pickle('grand_df_beta.pkl')

#df        = df[:200]
#%% generate data for CEBRA
import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra.datasets
import pandas as pd
from scipy.signal import hilbert

df['sin'] = None
df['cos'] = None

for row in df.iterrows():
    row = row[1]
    # compute the angle based on u and v
    u = row.u[0]
    v = row.v[0]
    angle = np.arctan2(v,u)

    # reshape to  256 x time
    angle = angle.reshape(256,-1)

    df.loc[row.name,'sin'] = [np.sin(angle)]
    df.loc[row.name,'cos'] = [np.cos(angle)]


#%%

trl_lengths = []
for i in range(len(df)):
    trl_length = np.arange(0,np.size(df.sin[i][0],1),1)
    trl_lengths.append(trl_length)
df['trial_length'] = trl_lengths


consonants = []
for i in range(len(df)):
    consonant = df.condition[i][0]
    consonants.append(consonant)

df['consonant'] = consonants

letter_reps  = []
binary_reps  = []
trialID_reps = []

for i in range(len(df)):
    rep_pre  = df.transition_time[i]
    rep_post = np.size(df.sin[i][0],1)-(df.transition_time[i])
    
    letter_rep = np.concatenate((np.repeat(df.consonant[i],rep_pre), np.repeat(df.vowel[i],rep_post)))
    letter_reps.append(letter_rep)
    binary_rep = np.concatenate((np.repeat(1,rep_pre), np.repeat(0,rep_post)))
    binary_reps.append(binary_rep)
    trialID_rep = np.repeat(i+1,rep_pre+rep_post)
    trialID_reps.append(trialID_rep)

df['letter_rep']  = letter_reps
df['binary_rep']  = binary_reps
df['trialID_rep'] = trialID_reps

# %% convert the data to a numpy array
ntrials = 200

for i in range(ntrials):
    if i == 0:
        sins     = np.array(df.sin[i][0])
        coses    = np.array(df.cos[i][0])
        letters   = np.array(df.letter_rep[i])
        binaries  = np.array(df.binary_rep[i])
        trial_lng = np.array(df.trial_length[i])
    else:
        sins = np.concatenate((sins,np.array(df.sin[i][0])),axis=1)
        coses = np.concatenate((coses,np.array(df.cos[i][0])),axis=1)
        letters = np.concatenate((letters,np.array(df.letter_rep[i])))
        binaries = np.concatenate((binaries,np.array(df.binary_rep[i])))
        trial_lng = np.concatenate((trial_lng,np.array(df.trial_length[i])))

letters   = letters[np.newaxis,:]
binaries  = binaries[np.newaxis,:]
trial_lng = trial_lng[np.newaxis,:]


# %%
np.save('sins.npy',sins)
np.save('coses.npy',coses)
np.save('binaries.npy',binaries)
np.save('trial_lng.npy',trial_lng)
np.save('letters.npy',letters)


# %%
