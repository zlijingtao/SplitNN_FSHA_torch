# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

import FSHA_torch
from datasets_torch import *

xpriv, xpub = load_mnist()

batch_size = 64
id_setup = 4
hparams = {
    'WGAN' : True,
    'gradient_penalty' : 500.,
    'style_loss' : None,
    'lr_f' :  0.00001,
    'lr_tilde' : 0.00001,
    'lr_D' : 0.0001,
}

fsha = FSHA_torch.FSHA(xpriv, xpub, id_setup-1, batch_size, hparams)

log_frequency = 500
LOG = fsha(10000, verbose=True, progress_bar=True, log_frequency=log_frequency)
# LOG = fsha(500, verbose=True, progress_bar=True, log_frequency=log_frequency)

def plot_log(ax, x, y, label):
    ax.plot(x, y, color='black')
    ax.set(title=label)
    ax.grid()

n = 4
fix, ax = plt.subplots(1, n, figsize=(n*5, 3))
x = np.arange(0, len(LOG)) * log_frequency 

plot_log(ax[0], x, LOG[:, 0], label='Loss $f$')
plot_log(ax[1], x, LOG[:, 1],  label='Loss $\\tilde{f}$ and $\\tilde{f}^{-1}$')
plot_log(ax[2], x, LOG[:, 2],  label='Loss $D$')
plot_log(ax[3], x, LOG[:, 3],  label='Reconstruction error (VALIDATION)')


n = 20
X = torch.from_numpy(getImagesDS(xpriv, n)).cuda()
X_recovered, control = fsha.attack(X)

X = X.detach().cpu().permute(0,2,3,1).numpy()
X_recovered = X_recovered.detach().cpu().permute(0,2,3,1).numpy()


def plot(X):
    n = len(X)
    X = (X+1)/2
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=-.05)
    for i in range(n):
        ax[i].imshow((X[i]), cmap='inferno');  
        ax[i].set(xticks=[], yticks=[])
        ax[i].set_aspect('equal')
        
    return fig


fig = plot(X)
fig = plot(X_recovered)
# %%
