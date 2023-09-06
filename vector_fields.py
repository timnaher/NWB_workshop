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
df = pd.read_pickle('df.pkl')

fig_lfp, ax_lfp     = plt.subplots()
fig_phase, ax_phase = plt.subplots()

ims_phase,ims_lfp  = [],[]

data = df.iloc[0].lfp
data = data.reshape(16,16,-1)
data = zscore_3d(data)
data = bandpass_filter_3d(lfp, 15, 25, Fs)


# %%
