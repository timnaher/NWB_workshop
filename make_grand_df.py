#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from IPython.display import display, clear_output
import time
import scipy.ndimage as ndimage

# load the data
sessions = ['B105','B15','B76','B8','B89','B9','B1']

grand_df = pd.DataFrame()
for ses in sessions:
    df = pd.read_pickle(f'df_{ses}vecfields_beta.pkl')
    grand_df = grand_df.append(df,ignore_index=True)

# save the grand df
grand_df.to_pickle('grand_df_beta.pkl')

# %%
