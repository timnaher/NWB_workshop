#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import *
# load the df

# loop over sessions
df = pd.DataFrame()
sessions = ['B105','B15','B76','B8','B89','B9','B1']
dandiset_id = "000019"

for ses in sessions:
    file_path   = f'sub-EC2/sub-EC2_ses-EC2-{ses}.nwb'
    nwbfile     = load_nwbfile(dandiset_id,file_path)
    electrodes  = nwbfile.electrodes.to_dataframe()
    df_sub = pd.read_pickle(f'df_{ses}vecfields_beta.pkl')
    df     = pd.concat([df, df_sub], ignore_index=True)


# get electrode labels

# convert to categorical
electrodes['location'] = electrodes.location.astype('category').cat.codes.values

electrode_labels = electrodes['location'].values.reshape(16,16,)
# convert the labels into numbers




#%%

df_subcond = df[df.condition=='raa']
# get all the lengths
lengths = []
for i,row in enumerate(df_subcond.iterrows()):
    row = row[1]
    lengths.append(row.lfp.shape[1])

# find minimum length
maxlength = np.max(lengths)
fig, ax = plt.subplots()
for i,row in enumerate(df_subcond.iterrows()):
    row = row[1]
    thistrial = row.lfp
    # subtract the mean
    thistrial = thistrial -  np.nanmean(thistrial,axis=1)[:,None]

    ax.plot(thistrial[0,:100])
    if thistrial.shape[1] < maxlength:
        pads      = np.ones((256,maxlength - thistrial.shape[1])) * np.nan
        thistrial = np.column_stack((thistrial,pads))
    erp = thistrial if i == 0 else np.nansum(np.dstack((erp,thistrial)),2)

grand_erp = erp/i




# %% Show the ERP for pre vs post

for j,row in enumerate(df.iterrows()):
    row          = row[1]
    U_grand_pre  = row.u_pre[0]  if j == 0 else np.nansum(np.dstack((U_grand_pre,row.u_pre[0])),  2)
    V_grand_pre  = row.v_pre[0]  if j == 0 else np.nansum(np.dstack((V_grand_pre,row.v_pre[0])),  2)
    U_grand_post = row.u_post[0] if j == 0 else np.nansum(np.dstack((U_grand_post,row.u_post[0])),2)
    V_grand_post = row.v_post[0] if j == 0 else np.nansum(np.dstack((V_grand_post,row.v_post[0])),2)

# normalize the vectors by deviding by the number of trials
U_grand_pre  = U_grand_pre/len(df)
V_grand_pre  = V_grand_pre/len(df)
U_grand_post = U_grand_post/len(df)
V_grand_post = V_grand_post/len(df)

# Compute magnitude for the vectors
magnitude_pre  = np.sqrt(U_grand_pre**2 + V_grand_pre**2)
magnitude_post = np.sqrt(U_grand_post**2 + V_grand_post**2)

# Normalize the vectors
U_grand_pre  /= magnitude_pre
V_grand_pre  /= magnitude_pre
U_grand_post /= magnitude_post
V_grand_post /= magnitude_post


# %% Figure 1: Plot the ERP for pre and post
fig, ax = plt.subplots(1,2, figsize=(10,5))
plt.set_cmap('Set2')

# rotate the electrode labels -90 degrees
roated_electrode_labels = np.fliplr(np.rot90(electrode_labels,1))

# take the electrode labels and plot them as a matrix underneath
ax[0].imshow(roated_electrode_labels)
#ax[0].quiver( U_grand_pre.T , V_grand_pre.T )
x = np.arange(0,16,1)
y = np.arange(0,16,1)
seed_points = np.array([[0, ], 
                        [14,]])

#ax[0].streamplot(x,y,U_grand_pre ,
# V_grand_pre , color='k')
ax[0].set_title('Vowel period')
ax[1].imshow(roated_electrode_labels)
ax[1].quiver( U_grand_post, V_grand_post)
ax[1].set_title('Consonant period')

# save the figure
#fig.savefig('erp_pre_post.pdf',dpi=300)

# %%
