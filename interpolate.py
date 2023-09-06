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
from scipy.signal import convolve2d

# load the data
df = pd.read_pickle('df.pkl')
Fs = 3051.7578

data = df.iloc[0].lfp
data = data.reshape(16,16,-1)

x = np.arange(0, 16, 1)
y = np.arange(0, 16, 1)
z = data[:,:,0]

# interpolate values with RegularGridInterpolator
import numpy as np
from scipy.interpolate import griddata

def interpolate_nan(grid):
    # Indices of the grid where values are NaN
    nan_indices = np.argwhere(np.isnan(grid))
    not_nan_indices = np.argwhere(~np.isnan(grid))
    not_nan_values = grid[~np.isnan(grid)]
    interpolated_values = griddata(not_nan_indices, not_nan_values, nan_indices, method='nearest')
    
    # Fill the interpolated values back into the grid
    for index, value in zip(nan_indices, interpolated_values):
        grid[tuple(index)] = value
    return grid





#%%
data = zscore_3d(data)
data = bandpass_filter_3d(data , 15, 25, Fs)



