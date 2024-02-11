#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from utils import *

# loop over sessions
dandiset_id = "000019"

df = pd.DataFrame()
ses = 'B105'
file_path   = f'sub-EC2/sub-EC2_ses-EC2-{ses}.nwb'
nwbfile     = load_nwbfile(dandiset_id,file_path)
electrodes  = nwbfile.electrodes.to_dataframe()
locations = electrodes.location.astype('category').cat.codes.values
locations_string = electrodes.location.values



# plot the electrode positions as scatter and color by locations
plt.set_cmap('tab20')
plt.scatter(electrodes.y,electrodes.z,c=locations)
# add the label as legend to the colors
plt.legend(['1','2','3'], loc='upper left', bbox_to_anchor=(1,1))



# %%
# make the location column in electrodes a category
electrodes['location'] = electrodes.location.astype('category')

ax1 = electrodes.plot.scatter(x='y',
                      y='z',
                      c='location', cmap='Set2',s=550)


ax1.set_xlabel('y')
ax1.set_ylabel('z')
# change the size of the figure
fig = plt.gcf()
fig.set_size_inches(15,10)
# set axis off
ax1.axis('off')

# save the figure
plt.savefig('electrode_locations.pdf',dpi=300,bbox_inches='tight')
# %%
