
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.stats import gaussian_kde 

class Visualizer: 
    """
    Trieda pre komplexnú 2D vizualizáciu latentného priestoru.
    """
    def __init__(self, analysis_files=[], figsize=(12, 3), nbins=40, cmap=plt.cm.jet):
        self.analysis_files = analysis_files
        self.figsize = figsize if figsize is not None else (12, 3)
        self.nbins = nbins
        self.cmap = cmap

    def make_visualization(self, lows_input, title="Latent Space Visualization"):
        if isinstance(lows_input, torch.Tensor):
            lows = lows_input.detach().cpu().numpy()
        elif isinstance(lows_input, np.ndarray):
            lows = lows_input
        else:
            return

        if lows.shape[1] < 2:
            print("[Visualizer] Latentný priestor musí mať aspoň 2 rozmery.")
            return

        x, y = lows[:, 0], lows[:, 1]
        
        ncols = 3
        nrows = math.ceil(len(self.analysis_files) / ncols) + 1
        fig_h = self.figsize[1] * nrows
        
        fig, axes = plt.subplots(
            ncols=ncols, 
            nrows=nrows,
            figsize=(self.figsize[0], fig_h), 
            sharex=True,
            sharey=True,
            squeeze=False
        )
        
        axes[0][0].set_title(f"Scatterplot ({title})", fontsize=10)
        axes[0][0].scatter(x, y, s=0.1, cmap=self.cmap)

        axes[0][1].set_title("2D Histogram", fontsize=10)
        hist2d = axes[0][1].hist2d(x, y, bins=self.nbins, cmap=self.cmap)
        fig.colorbar(hist2d[3], ax=axes[0][1], label='Frekvencia')

        try:
            k = gaussian_kde((x, y))
            xi, yi = np.mgrid[x.min():x.max():self.nbins * 1j, y.min():y.max():self.nbins * 1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            axes[0][2].set_title("2D Density with shading", fontsize=10)
            
            density_plot = axes[0][2].pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap=self.cmap)
            fig.colorbar(density_plot, ax=axes[0][2], label='Hustota')

        except np.linalg.LinAlgError:
            axes[0][2].set_title("2D Density (Failed)", fontsize=10)
            
        plt.tight_layout()
        plt.show()
        plt.pause(0.01)