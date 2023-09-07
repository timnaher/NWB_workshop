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
df = pd.read_pickle('grand_df_beta.pkl')

df_subcond = df[df.condition=='raa']
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
    erp = thistrial if i == 0 else np.nansum(np.dstack((erp,thistrial)),2)

grand_erp = erp/i



#%%
#filter the grand erp in the beta band
Fs       = 3051.7578
band = 'beta'
if band == 'beta':
    low_bound, up_bound = 12, 20
elif band == 'alpha':
    low_bound, up_bound = 7, 11

grand_erp = bandpass_filter_3d(grand_erp,low_bound,up_bound,Fs)
# %%
# reshape the data
grand_erp = grand_erp.reshape(16,16,-1)
plot_movie(grand_erp,name='erp_raa')


# %%
