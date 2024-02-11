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
from scipy.signal import hilbert
from utils import *

# load the data
df = pd.read_pickle('df_B1_beta.pkl')
fig_lfp, ax_lfp     = plt.subplots()
fig_phase, ax_phase = plt.subplots()
fig_vec, ax_vec     = plt.subplots()
ims_phase,ims_lfp, ims_vec   = [],[],[]


data = df.iloc[0].lfp.reshape(16,16,-1)
data = zscore_3d(data)
vmin = np.nanmin(data)
vmax = np.nanmax(data)

#%% LFP PHASE

for i in range(data.shape[2]):  # 100 frames
    # smooth the current matrix frame
    frame = data[:,:,i]
    phase = np.angle(hilbert(frame))

    #frame = ndimage.gaussian_filter(frame, sigma=0.5)
    im_phase = ax_phase.imshow(phase,vmin=-np.pi, vmax=np.pi,cmap='twilight')
    # add x and y labels

    ims_phase.append([im_phase])

    im_lfp   = ax_lfp.imshow(frame,vmin=vmin,vmax=vmax,cmap='viridis')

    ims_lfp.append([im_lfp])


ani_lfp   = animation.ArtistAnimation(fig_lfp, ims_lfp, interval=50, blit=True,repeat_delay=500)
ani_phase = animation.ArtistAnimation(fig_phase, ims_phase, interval=50, blit=True,repeat_delay=500)

writer = PillowWriter(fps=50)

ani_lfp.save(    "lfp_demo_8-12.gif", writer=writer)
ani_phase.save("phase_demo_8-12.gif", writer=writer)


#%% VECTOR FIELDS
# load the data
df = pd.read_pickle('df_B1_beta.pkl')
fig_vec, ax_vec     = plt.subplots()

ims_vec   = []






#%%

def plot_movie(data,name):
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    for i in range(data.shape[2]):  # 100 frames
        # smooth the current matrix frame
        frame = data[:,:,i]
        phase = np.angle(hilbert(frame))

        #frame = ndimage.gaussian_filter(frame, sigma=0.5)
        im_phase = ax_phase.imshow(phase,vmin=-np.pi, vmax=np.pi,cmap='twilight')
        ims_phase.append([im_phase])

        im_lfp   = ax_lfp.imshow(frame,vmin=vmin,vmax=vmax,cmap='viridis')
        ims_lfp.append([im_lfp])


    ani_lfp   = animation.ArtistAnimation(fig_lfp, ims_lfp, interval=50, blit=True,repeat_delay=500)
    ani_phase = animation.ArtistAnimation(fig_phase, ims_phase, interval=50, blit=True,repeat_delay=500)

    writer = PillowWriter(fps=50)

    ani_lfp.save(    f"lfp_{name}.gif", writer=writer)
    ani_phase.save(f"phase_{name}.gif", writer=writer)

