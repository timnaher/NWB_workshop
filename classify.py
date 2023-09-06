#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the df 'df_vecfields.pkl'
df = pd.read_pickle('df_vecfields.pkl')


# append the u_post and v_post  and get rid of the dataframe strucutre
for itrial in np.arange(df.shape[0]):
    upost_flat = df.iloc[itrial].u_post[0].flatten()
    vpost_flat = df.iloc[itrial].v_post[0].flatten()
    divergence = df.iloc[itrial].div_post[0].flatten()
    curl       = df.iloc[itrial].curl_post[0].flatten()
    # concatenate
    #feat = np.concatenate((upost_flat,vpost_flat))
    feat = divergence
    X = feat if itrial == 0 else np.row_stack((X,feat))


# scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['id'].values

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#% use a KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# train the model
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# predict
y_pred = neigh.predict(X_test)

# get the accuracy
accuracy_score(y_test, y_pred)

#%%
# train the model
clf = svm.LinearSVC()
clf.fit(X_train,y_train)

# predict
y_pred = clf.predict(X_test)

# get the accuracy
accuracy_score(y_test, y_pred)


#%% use a KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# train the model
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# predict
y_pred = neigh.predict(X_test)

# get the accuracy
accuracy_score(y_test, y_pred)

#%% use a RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# train the model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)

# get the accuracy
accuracy_score(y_test, y_pred)

#%% use a GaussianNB
from sklearn.naive_bayes import GaussianNB

# train the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# predict
y_pred = gnb.predict(X_test)

# get the accuracy
accuracy_score(y_test, y_pred)



# %% use a neural network
from sklearn.neural_network import MLPClassifier

# train the model
clf = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)

# get the accuracy
accuracy_score(y_test, y_pred)

# %%
