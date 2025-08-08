import numpy as np
import torch
import umap
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from src.setup import neurotransmitters, embeddings_path

if os.path.exists(os.path.join(embeddings_path, 'LABELLED_PSD.pt')) and os.path.exists(os.path.join(embeddings_path, 'LABELLED_REST.pt')):
    LABELLED_PSD = torch.load(os.path.join(embeddings_path, 'LABELLED_PSD.pt'))
    #LABELLED_REST = torch.load(os.path.join(embeddings_path, 'LABELLED_REST.pt')) # TODO: Check if this is necessary
    print('Embeddings found and loaded')
else:
    from .dataset import LABELLED_PSD, LABELLED_REST
    print('Embeddings not found, loading from dataset')

print(f'LABELLED_PSD: {len(LABELLED_PSD)}')
print(f'LABELLED_REST: {len(LABELLED_REST)}') if 'LABELLED_REST' in locals() or 'LABELLED_REST' in globals() else None

def main():
    reducer = umap.UMAP(random_state=42)
            
    display_total = False
    if display_total:

        # Load and reshape embeddings (same as original approach but more memory-aware)
        psd_embeddings, psd_labels = zip(*LABELLED_PSD)
        rest_embeddings, rest_labels = zip(*LABELLED_REST)
        
        _embeddings = [psd_embeddings, rest_embeddings]
        _labels = [psd_labels, rest_labels]
        
        embeddings = []
        for e in _embeddings:
            embeddings.extend(e)
        embeddings = np.array(embeddings)
        
        labels = []
        for l in _labels:
            labels.extend(l)
        labels = np.array(labels)

    else:
        psd_embeddings, _ = zip(*LABELLED_PSD)
        embeddings = np.array(psd_embeddings)
        
        
        nb_images_per_class = len(embeddings) // len(neurotransmitters)
        psd_labels = [[i] * nb_images_per_class for i in range(len(neurotransmitters))]
        labels = np.hstack(psd_labels)
                
    # Get dimensions without loading everything into memory
    sample_embedding = embeddings[0][0] if isinstance(embeddings[0], list) else embeddings[0]
    feat_dim = sample_embedding.shape[-1]

    # Count total embeddings
    total_embeddings = sum(len(e) if isinstance(e, list) else 1 for e in embeddings)
    print(f'Total embeddings: {total_embeddings}, Feature dim: {feat_dim}')


    umap_all = reducer.fit_transform(embeddings)



    unique_labels = np.unique(labels)
    if unique_labels.shape[0] == 6:    
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink'] 
        markers = ['o', 'x', 'v', '^', 's', 'p']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        sizes = [1, 1, 1, 1, 1, 1]
        names = neurotransmitters

    elif unique_labels.shape[0] == 2:
        colors = ['blue', 'red'] 
        markers = ['o', 'x'] 
        alphas = [0.5, 1.0]
        sizes = [1, 3]
        names = ['Rest', 'PSD']

    for i, label_val in enumerate(unique_labels):
        mask = labels == label_val
        plt.scatter(umap_all[mask, 0], umap_all[mask, 1],
                alpha=alphas[i], s=sizes[i], c=colors[i], 
                marker=markers[i], label=names[i])

    plt.title(f'PSD neurotransmitter classes UMAP visualization') if unique_labels.shape[0] == 6 else plt.title(f'Entire dataset UMAP visualization')
    ax = plt.gca()
    ax.set_aspect(0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    main()