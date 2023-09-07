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
from utils import *
import numpy as np
from scipy.signal import convolve2d, hilbert


sessions = ['B105']#,'B15','B76','B8','B89','B9','B1']
Fs       = 3051.7578


for ses in sessions:
    print(ses)
    # load the data
    df = pd.read_pickle(f'df_{ses}_beta.pkl')

    # make a new column for the amplitude envelope
    df['amplitude_envelope'] = None

    # loop over trials
    for row in df.iterrows():
        row = row[1]

        # compute the amplitude envelope over the whole trial
        thisLFP = row.lfp
        analytic_signal = hilbert(thisLFP)
        amplitude_envelope = np.abs(analytic_signal)

        # attach the envelope to the dataframe
        df.loc[row.name,'amplitude_envelope'] = [amplitude_envelope]



# %%

