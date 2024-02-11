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
    df
    df     = pd.concat([df, df_sub], ignore_index=True)



# convert to categorical
electrodes['location'] = electrodes.location.astype('category').cat.codes.values

electodes_labels = electrodes['location'].values.reshape(16,16,)

# flip lr
electodes_labels = np.fliplr(electodes_labels)
#electrode_labels = electrodes['location'].values.reshape(16,16,)
# convert the labels into numbers

# %%
df_consistency = pd.DataFrame(columns=[ '1_mean_dir',
                                        '1_r',
                                        '2_mean_dir',
                                        '2_r',
                                        '3_mean_dir',
                                        '3_r',
                                        '4_mean_dir',
                                        '4_r',
                                        '5_mean_dir',
                                        '5_r',
                                        '6_mean_dir',
                                        '6_r',
                                        '0_mean_dir',
                                        '0_r',
                                        'condition']) # 'condition
                                        



for row in df.iterrows():
    row = row[1]
    u = row.u[0]
    v = row.v[0]

    # get condition
    condition = row.condition

    # compute the angle based on u and v
    angles = np.arctan2(v, u)

    # find all x and y positions in electrodes_labels that are 1
    temp_data = []
    for area in np.unique(electodes_labels):
        x, y = np.where(electodes_labels == area)

        # collect all the angles at these positions
        angles_at_area = angles[x, y, :]

        # compute the circular mean of all angles at this area
        circ_mean = np.angle(np.mean(np.exp(1j * angles_at_area)))

        # compute the vector length
        vector_length = np.abs(np.mean(np.exp(1j * angles_at_area)))

        # save in a temporary list
        temp_data.append({ f'{area}_mean_dir': circ_mean,
                           f'{area}_r': vector_length,
                           'condition': condition})

    merged_dict = {}
    for d in temp_data:
        merged_dict.update(d)

    # Convert the list of dictionaries to DataFrame and concatenate
    df_consistency = pd.concat([df_consistency, pd.DataFrame([merged_dict])], ignore_index=True)


# %%
# Create polar histograms
fig, ax = plt.subplots(3, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 10))

# List of axes for easy iteration
axes = ax.ravel()

# Only take the first 6 columns for plotting
for i, column in enumerate(df_consistency.columns[::2]):
    axes[i].hist(df_consistency[column].values, bins=32, color='#888888', alpha=0.7) # muted color
    axes[i].set_yticklabels([])  # remove radial tick labels
    axes[i].set_rlabel_position(-180)  # move radial labels out of view
    axes[i].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])  # set tick labels in radians
    # plot a line at the circular mean
    #axes[i].spines['polar'].set_visible(False)  # remove the outer circle line for a cleaner look
    #axes[i].grid(False)  # remove grid

plt.tight_layout()
plt.show()

plt.show()

fix, ax = plt.subplots(3,2)
ax[0,0].hist( df_consistency['0_r'].values, bins=30)
ax[0,1].hist( df_consistency['1_r'].values, bins=30)
ax[1,0].hist( df_consistency['2_r'].values, bins=30)
ax[1,1].hist( df_consistency['3_r'].values, bins=30)
ax[2,0].hist( df_consistency['4_r'].values, bins=30)
ax[2,1].hist( df_consistency['5_r'].values, bins=30)




# %%
# classify the area based on the vector length

# import train_test_split
from sklearn.model_selection import train_test_split

# get the data
X = df_consistency[['0_r','1_r','2_r','3_r','4_r','5_r']].values

# select the directions
#X = df_consistency[['0_mean_dir','1_mean_dir','2_mean_dir','3_mean_dir','4_mean_dir','5_mean_dir']].values
# take sin and cos
#X = np.column_stack((np.sin(X),np.cos(X)))

y = df_consistency['condition'].astype('category').cat.codes.values

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# import the classifier
from sklearn.svm import SVC

# create the classifier
clf = SVC(kernel='linear')

# fit the classifier
clf.fit(X_train, y_train)

# predict the labels
y_pred = clf.predict(X_test)

# compute the accuracy
accuracy = accuracy_score(y_test, y_pred)

# print the accuracy
print(f'The accuracy of the classifier is {accuracy}')


# %%
