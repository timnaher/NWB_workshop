#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_pickle('grand_df_beta.pkl')


# %%
