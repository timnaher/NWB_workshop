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
from tqdm.notebook import tqdm  # if you're in a Jupyter Notebook

def frameByFrameHornSchunck(im1, im2, alpha=1, ite=100, uInitial=None, vInitial=None):
    if uInitial is None:
        uInitial = np.zeros(im1.shape)
    if vInitial is None:
        vInitial = np.zeros(im2.shape)
    u = uInitial
    v = vInitial

    fx, fy, ft = computeDerivatives(im1, im2)

    kernel_1 = np.array([[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]])

    for _ in range(ite):
        uAvg = convolve2d(u, kernel_1, mode='same')
        vAvg = convolve2d(v, kernel_1, mode='same')
        u = uAvg - (fx * ((fx * uAvg) + (fy * vAvg) + ft)) / (alpha**2 + fx**2 + fy**2)
        v = vAvg - (fy * ((fx * uAvg) + (fy * vAvg) + ft)) / (alpha**2 + fx**2 + fy**2)
    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0
    return u, v

def computeDerivatives(im1, im2):
    if im2.size == 0:
        im2 = np.zeros(im1.shape)
    
    I = im1 + im2
    kernel_fx = np.array([-1, 8, 0, -8, 1]).reshape(1, -1) * (1/12)  # reshaped to (1, 5)
    kernel_fy = np.array([-1, 8, 0, -8, 1]).reshape(-1, 1) * (1/12)  # reshaped to (5, 1)
    
    fx = convolve2d(I/2, kernel_fx, mode='same')
    fy = convolve2d(I/2, kernel_fy, mode='same')
    ft = convolve2d(im1, 0.25*np.ones((2,2)), mode='same') + convolve2d(im2, -0.25*np.ones((2,2)), mode='same')
    
    fx = -fx
    fy = -fy
    
    return fx, fy, ft

def opticalFlowHS(data, alpha=1, max_iter=100, wait_bar=True):
    rows, cols, frames = data.shape
    u = np.zeros((rows, cols, frames - 1))
    v = np.zeros((rows, cols, frames - 1))
    for frame in range(frames - 1):
        u[:, :, frame], v[:, :, frame] = frameByFrameHornSchunck(data[:, :, frame], data[:, :, frame+1], alpha, max_iter)
        
        if wait_bar:
            print(f"Computing Optical Flow Vector Fields: {int((frame/(frames-1))*100)}% complete")
    if wait_bar:
        print("Optical Flow computation complete!")

    x, y = np.meshgrid(range(cols), range(rows))
    return x, y, u, v

#%%

sessions = ['B105','B15','B76','B8','B89','B9','B1']
Fs       = 3051.7578
band = 'beta'


for ses in sessions:
    print(ses)
    # load the data
    df = pd.read_pickle(f'df_{ses}_{band}.pkl')

    df['u'] = None
    df['v'] = None
    for j in tqdm(range(len(df))):
        data = zscore_3d(df.iloc[j].lfp.reshape(16,16,-1))
        x, y, u, v = opticalFlowHS(data, alpha=1, max_iter=100, wait_bar=False)

        # save u and v in the dataframe
        df.loc[j, 'u'] = [u]
        df.loc[j, 'v'] = [v]


    df['u_pre']     = None
    df['v_pre']     = None
    df['u_post']    = None
    df['v_post']    = None
    df['div_pre']   = None
    df['div_post']  = None
    df['curl_pre']  = None
    df['curl_post'] = None

    for row in tqdm(df.iterrows()):
        row = row[1]

        # compute the pre and post transition vector fields
        u_pre  = np.nanmean(row.u[0][:,:,:row.transition_time],axis=2)
        v_pre  = np.nanmean(row.v[0][:,:,:row.transition_time],axis=2)
        u_post = np.nanmean(row.u[0][:,:,row.transition_time:],axis=2)
        v_post = np.nanmean(row.v[0][:,:,row.transition_time:],axis=2)

        # compute the divergence of the vector fields
        div_pre  = np.gradient(u_pre)[0]  + np.gradient(v_pre)[1]
        div_post = np.gradient(u_post)[0] + np.gradient(v_post)[1]

        # compute the curl
        curl_pre  = np.gradient(u_pre)[1]  - np.gradient(v_pre)[0]
        curl_post = np.gradient(u_post)[1] - np.gradient(v_post)[0]

        # append to the dataframe
        df.loc[row.name,'u_pre']     = [u_pre]
        df.loc[row.name,'v_pre']     = [v_pre]
        df.loc[row.name,'u_post']    = [u_post]
        df.loc[row.name,'v_post']    = [v_post]

        df.loc[row.name,'div_pre']   = [div_pre]
        df.loc[row.name,'div_post']  = [div_post]

        df.loc[row.name,'curl_pre']  = [curl_pre]
        df.loc[row.name,'curl_post'] = [curl_post]


    df.to_pickle(f'df_{ses}vecfields_{band}.pkl')


# %%
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

fig_lfp, ax_lfp     = plt.subplots()
fig_phase, ax_phase = plt.subplots()
fig_vec, ax_vec     = plt.subplots()

ims_phase,ims_lfp,ims_vec   = [],[],[]


vmin = np.nanmin(data)
vmax = np.nanmax(data)

for i in range(data.shape[2]-1):  # 100 frames
    print(i)
    # smooth the current matrix frame
    frame = data[:,:,i]
    phase = np.angle(hilbert(frame))

    #frame = ndimage.gaussian_filter(frame, sigma=0.5)
    im_phase = ax_phase.imshow(phase,vmin=-np.pi, vmax=np.pi,cmap='twilight')
    ims_phase.append([im_phase])

    im_lfp   = ax_lfp.imshow(frame,vmin=vmin,vmax=vmax,cmap='viridis')
    ims_lfp.append([im_lfp])

    im_vec   = ax_vec.quiver(x,y,u[:,:,i],v[:,:,i])
    ims_vec.append([im_vec])


#ani_lfp   = animation.ArtistAnimation(fig_lfp, ims_lfp, interval=50, blit=True,repeat_delay=500)
#ani_phase = animation.ArtistAnimation(fig_phase, ims_phase, interval=50, blit=True,repeat_delay=500)
ani_vec   = animation.ArtistAnimation(fig_vec, ims_vec, interval=50, blit=True,repeat_delay=500)
writer    = PillowWriter(fps=50)

#ani_lfp.save(    "lfp_demo_8-12.gif",   writer=writer)
#ani_phase.save(  "phase_demo_8-12.gif", writer=writer)
ani_vec.save(    "vec_demo_8-12.gif",   writer=writer)

# %%
