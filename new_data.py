# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra.datasets
from cebra import CEBRA
import pandas as pd
from scipy.signal import hilbert

#df = pd.read_pickle('df_B105_beta.pkl')
df = pd.read_pickle('df_B1_beta.pkl')

#%% make the amplitude envelope
df['amplitude_envelope'] = None
for row in df.iterrows():
    row = row[1]
    # compute the amplitude envelope over the whole trial
    thisLFP = row.lfp
    analytic_signal = hilbert(thisLFP)
    amplitude_envelope = np.abs(analytic_signal[:,30:-30])

    # attach the envelope to the dataframe
    df.loc[row.name,'amplitude_envelope'] = [amplitude_envelope]



#%%

trl_lengths = []
for i in range(len(df)):
    trl_length = np.arange(0,np.size(df.amplitude_envelope[i][0],1),1)
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
    rep_post = np.size(df.amplitude_envelope[i][0],1)-(df.transition_time[i])
    
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
ntrials = 50

for i in range(ntrials):
    if i == 0:
        lfps      = np.array(df.amplitude_envelope[i][0])
        letters   = np.array(df.letter_rep[i])
        binaries  = np.array(df.binary_rep[i])
        trial_lng = np.array(df.trial_length[i])
    else:
        lfps = np.concatenate((lfps,np.array(df.amplitude_envelope[i][0])),axis=1)
        letters = np.concatenate((letters,np.array(df.letter_rep[i])))
        binaries = np.concatenate((binaries,np.array(df.binary_rep[i])))
        trial_lng = np.concatenate((trial_lng,np.array(df.trial_length[i])))

letters   = letters[np.newaxis,:]
binaries  = binaries[np.newaxis,:]
trial_lng = trial_lng[np.newaxis,:]

lfp_array = np.concatenate((lfps,binaries,trial_lng),axis=0).T

# %% HYBRID cebra model (neural + behavioral)
max_iterations = 500

cebra_test_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-3,
                        temperature=1,
                        output_dimension=32,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid = False)

# %%

amplitude_data        = lfp_array[:,0:256]
consonant_data  = lfp_array[:,256].astype(int)
timestamps_data = lfp_array[:,257]

np.save("amplitude_data.npy", lfp_data)
np.save("consonant_data.npy", consonant_data)
np.save("timestamps_data.npy", timestamps_data)

#%%
cebra_test_model.fit(lfp_data,timestamps_data)
cebra_test_model.save("test_model.pt")

cebra.plot_loss(cebra_test_model)

# %%

# CEBRA-Hybrid
#cebra_hybrid_model = cebra.CEBRA.load("cebra_hybrid_model.pt")
cebra_test= cebra_test_model.transform(lfp_data)

#%%
# CEBRA-Behavior with shuffled labels
#cebra_behavior_shuffled_model = cebra.CEBRA.load("cebra_behavior_shuffled_model.pt")
#cebra_behavior_shuffled = cebra_behavior_shuffled_model.transform(hippocampus_pos.neural)

# %%

fig = plt.figure(figsize=(10,10))
cmap = plt.cm.get_cmap('viridis', 10)
ax=cebra.plot_embedding(embedding=cebra_test, embedding_labels=consonant_data, cmap=cmap)

# change view on 3d plot
plt.show()


# %%
np.save("lfp_array.npy", lfp_array)
# %%