import numpy as np
import matplotlib as mpl

def color_afmhot(val, vmin, vmax):
    v = (val - vmin) / (vmax - vmin)

    if v < 0.625:
        R = np.floor(v * 256 / 0.625)
        G = np.floor(v * 256)
        return (int(R), int(G), 0)
    
    elif v < 0.75:
        u = (v - 0.625)
        if u < 0.0:
            return (255, 159, 0)
        elif u >= 0.125:
            return (255, 191, 0)
        G = np.floor(v * 256)
        return (255, int(G), 0)
    
    else:
        u = (v - 0.75)
        if u < 0.0:
            return (255, 191, 0)
        elif u >= 0.25:
            return (255, 255, 255)
        B = np.floor(u * 256 / 0.25)
        G = np.floor(v * 256)
        return (255, int(G), int(B))

def array_to_rgb(arr, vmin=None, vmax=None):
    arr = np.asarray(arr)
    if vmin is None: vmin = arr.min()
    if vmax is None: vmax = arr.max()

    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            rgb[i, j] = color_afmhot(arr[i, j], vmin, vmax)
    return rgb

def create_afmhot_cmap(vmin, vmax):
    colors = []
    for i in range(256):
        val = vmin + (vmax - vmin) * (i / 255)
        colors.append(np.array(color_afmhot(val, vmin, vmax))/255)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("afmhot_custom_heights", colors, N=256)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return cmap, norm
