import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


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