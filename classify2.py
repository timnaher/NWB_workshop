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

# load the data
df = pd.read_pickle('grand_df_beta.pkl')

#%% generate features
# here we want to see if we can classify whether a vector field is pre or post transition

for j,row in enumerate(df.iterrows()):
    row = row[1]

    for i,loc in enumerate(['pre','post']):
        u = row[f'u_{loc}'][0]
        v = row[f'v_{loc}'][0]

        # compute the angle based on u and v
        angle = np.arctan2(v,u)

        divergence = row[f'div_{loc}'][0]
        curl       = row[f'curl_{loc}'][0]

        # concatenate
        feat = np.concatenate(((np.sin(angle)).flatten(),
                               np.cos(angle).flatten()))
                               #divergence.flatten(),
                               #curl.flatten()))
        #feat = np.concatenate((divergence.flatten(),curl.flatten()))
        #feat = u.flatten()
        X = feat if j == 0 else np.row_stack((X,feat))
        y = np.array([i]) if j == 0 else np.row_stack((y,np.array([i])))
    


# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Your actual model
model             = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.ravel())  # Use ravel() here

y_pred            = model.predict(X_test)
observed_accuracy = accuracy_score(y_test, y_pred)

# Permutation test
n_permutations      = 100
count               = 0
permuted_accuracies = []  # List to store accuracies from each permutation

for jj in range(n_permutations):
    print(jj)
    # Permute the labels
    y_train_permuted = np.random.permutation(y_train)
    
    # Train the classifier on the permuted labels
    model = SVC(kernel='linear')
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
np.save('classifier/results/log_reg_p_value.npy',p_value)
np.save('classifier/results/log_reg_permuted_accuracies.npy',permuted_accuracies)
np.save('classifier/results/observed_accuracy.npy',observed_accuracy)

print(f"Observed Accuracy: {observed_accuracy}")
print(f"P-value: {p_value}")

# Plot histogram
plt.hist(permuted_accuracies, bins=30, alpha=0.7, label='Permuted Accuracies')
plt.axvline(observed_accuracy, color='red', linestyle='dashed', linewidth=2, label='Observed Accuracy')
plt.title('Permutation Test for Classifier Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# %%
