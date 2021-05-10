from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

features = np.load('tsne_results/features.npy').astype(np.float32)
data = np.load('tsne_results/data.npy').astype(np.float32)
domain = np.load('tsne_results/domain.npy').astype(np.float32)

dim, dim2, dim3, dim4 = features.shape
features = features.reshape((dim, dim2 * dim3 * dim4))
print('featues shape: ', features.shape)
print('data shape: ', data.shape)
domain = np.squeeze(domain, axis=1)
print('domain shape: ', domain.shape)
pca_features = PCA(50).fit_transform(features)
perplexity = [10, 15, 20, 25, 30, 35, 40, 45, 50]
for perp in perplexity:
    tsne_features = TSNE(perplexity=perp).fit_transform(
        pca_features
    )  #check out the TSNE  QA website, try perplexity 5~50, and do PCA
    if tsne_features.any():
        colours = ListedColormap(['#3DB7E9', '#F748A5', '#E69F00'])
        classes = ['Utrecht', 'Amsterdam', 'Singapore']
        scatter = plt.scatter(tsne_features[:, 0],
                              tsne_features[:, 1],
                              c=domain,
                              s=3,
                              cmap=colours)
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=classes,
                   prop={"size": 13})
        plt.show()
        plt.savefig(f'tsne_results/tsne_plot_perp{perp}.png')
        plt.close()
    #p in [5, 10, 15, 20, 35, 40, 50]