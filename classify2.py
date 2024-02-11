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
sessions = ['B105','B15','B76','B8','B89']#,'B9','B1']

for ses in sessions:
    file_path   = f'sub-EC2/sub-EC2_ses-EC2-{ses}.nwb'
    nwbfile     = load_nwbfile(dandiset_id,file_path)
    electrodes  = nwbfile.electrodes.to_dataframe()
    df_sub = pd.read_pickle(f'df_{ses}vecfields_alpha.pkl')
    df     = pd.concat([df, df_sub], ignore_index=True)


#df = pd.read_pickle('df_B105vecfields_beta.pkl')
print('done loading data')

#%% generate features
# here we want to see if we can classify whether a vector field is pre or post transition
for j,row in enumerate(df.iterrows()):
    row        = row[1]
    transition = row['transition_time']
    for i,loc in enumerate(['pre','post']):
        U_pre  = row.u[0][:,:,:transition]
        V_pre  = row.v[0][:,:,:transition]
        U_post = row.u[0][:,:, transition:]
        V_post = row.v[0][:,:, transition:]

        u = np.nanmean(U_pre,axis=2) if loc == 'pre' else np.nanmean(U_post,axis=2)
        v = np.nanmean(V_pre,axis=2) if loc == 'pre' else np.nanmean(V_post,axis=2)

        # compute the angle based on u and v
        angle      = np.arctan2(v,u)
        divergence = row[f'div_{loc}'][0]
        curl       = row[f'curl_{loc}'][0]

        # concatenate
        feat = np.concatenate(((np.sin(angle)).flatten(),
                               np.cos(angle).flatten()))
        if np.any(np.isnan(feat)):
            continue
        else:
            X = feat if j == 0 else np.row_stack((X,feat))
            y = np.array([i]) if j == 0 else np.row_stack((y,np.array([i])))
    
# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
print('start classification')

# Your actual model
model             = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.ravel())  # Use ravel() here

y_pred            = model.predict(X_test)
observed_accuracy = accuracy_score(y_test, y_pred)
print(f"Observed Accuracy: {observed_accuracy}")

# Permutation test
n_permutations      = 2000
count               = 0
permuted_accuracies = []  # List to store accuracies from each permutation

#%%
print('starting permutation test')
for jj in range(n_permutations):
    print(jj)
    # Permute the labels
    y_train_permuted = np.random.permutation(y_train)
    
    # Train the classifier on the permuted labels
    model             = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train_permuted.ravel())  # Use ravel() here
    
    # Test the classifier
    y_pred_permuted   = model.predict(X_test)
    permuted_accuracy = accuracy_score(y_test, y_pred_permuted)
    permuted_accuracies.append(permuted_accuracy)  # Save the permuted accuracy
    
    # Check if permuted accuracy is greater than or equal to the observed accuracy
    if permuted_accuracy >= observed_accuracy:
        count += 1

p_value = count / n_permutations

# save p_value and permuted accuracies to disk
#np.save('classifier/results/log_reg_p_value.npy',p_value)
#np.save('classifier/results/log_reg_permuted_accuracies.npy',permuted_accuracies)
#np.save('classifier/results/observed_accuracy.npy',observed_accuracy)

print(f"Observed Accuracy: {observed_accuracy}")
print(f"P-value: {p_value}")

#%% Plot histogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

fig, ax = plt.subplots()
# Plot the histogram on the given axis
ax.hist(permuted_accuracies, bins=50, alpha=0.7, label='Permuted Accuracies', density=True)

# Fit a normal distribution to the permutation accuracies
mu, std = norm.fit(permuted_accuracies)

xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

# Plot the fitted distribution and observed accuracy on the given axis
ax.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
ax.axvline(observed_accuracy, color='red', linestyle='dashed', linewidth=2, label='Observed Accuracy')

ax.set_title('Permutation Test for Classifier Accuracy')
ax.set_xlabel('Accuracy')
ax.set_ylabel('Frequency')
ax.legend()

# Save plot to disk first, then show
plt.savefig('classifier/results/log_reg_permutation_test_ALPHA.pdf')
plt.show()


# %%
