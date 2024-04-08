# this is a demo python code for visualizing one multi-channel protein image (save as .png)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

# ---------- specify these args ----------
img_id = '1'
path_img = f'train/images/{img_id}.tiff' # your path of the image of choice
path_protein_names = 'train/protein_names.csv' # your path of protein_names.csv
n_proteins = 52 # number of proteins 
# ----------------------------------------

img = imageio.v2.imread(path_img) # read the image
pts = pd.read_csv(path_protein_names) # read the protein names

nc = 5
nr = int(np.ceil(n_proteins/nc))
ss = 3
fig = plt.figure(figsize=(nc*ss, nr*ss), dpi=150)
gs = fig.add_gridspec(nr, nc,
                      width_ratios=[1]*nc,
                      height_ratios=[1]*nr)
gs.update(wspace=0.5, hspace=0.3)

for i in range(n_proteins):
    ax = plt.subplot(gs[i//nc, i%nc])
    im = img[i,:,:]
    ax.imshow(im, cmap = 'Greys', vmin=np.quantile(im, 0.05), vmax=np.quantile(im, 0.95))
    ax.set_title(f'{pts["Metal Tag"][i]}\n{pts["Target"][i]}')
    ax.set_aspect('equal', adjustable='box')
    # ax.axis('off')
plt.savefig(f'{img_id}.png')