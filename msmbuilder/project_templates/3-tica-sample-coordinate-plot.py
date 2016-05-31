"""Plot the result of sampling a tICA coordinate

{{header}}
"""

from subprocess import run

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from msmbuilder.io import load_trajs, load_generic

sns.set_style('ticks')
colors = sns.color_palette()

## Load
meta, ttrajs = load_trajs('ttrajs')
txx = np.concatenate(list(ttrajs.values()))

inds = load_generic("tica-dimension-0-inds.pickl")
straj = []
for traj_i, frame_i in inds:
    straj += [ttrajs[traj_i][frame_i, :]]
straj = np.asarray(straj)


## Overlay sampled trajectory on histogram
def plot_sampled_traj(ax):
    ax.hexbin(txx[:, 0], txx[:, 1],
              cmap='magma_r',
              mincnt=1,
              bins='log',
              alpha=0.8,
              )

    ax.plot(straj[:, 0], straj[:, 1], 'o-', label='Sampled')

    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)
    ax.legend(loc='best')


## Plot
fig, ax = plt.subplots(figsize=(7, 5))
plot_sampled_traj(ax)
fig.tight_layout()
fig.savefig('tica-dimension-0-heatmap.pdf')
run(['xdg-open', 'tica-dimension-0-heatmap.pdf'])
