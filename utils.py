import numpy as np
import json

def smooth_angles(angles, window=5):
    angles = np.asarray(angles)
    if angles.ndim == 1:
        return np.convolve(angles, np.ones(window) / window, mode='same')
    elif angles.ndim == 2:
        smoothed = np.zeros_like(angles)
        for i in range(angles.shape[0]):
            smoothed[i] = np.convolve(angles[i], np.ones(window) / window, mode='same')
        return smoothed
    else:
        raise ValueError("Input angles must be 1D or 2D array")

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)