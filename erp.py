#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load the df
df = pd.read_pickle('grand_df.pkl')

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
    if i == 0:
        erp = thistrial
    else:
        erp = np.nansum(np.dstack((erp,thistrial)),2)


grand_erp = erp/i
plt.plot(grand_erp[:10,:].T)
# %%
