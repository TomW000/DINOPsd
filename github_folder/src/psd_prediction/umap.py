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
#print(f'LABELLED_REST: {len(LABELLED_REST)}')


class LargeEmbeddingsProcessor:
    def __init__(self, batch_size=10000, n_components_pca=100, random_state=42):
        self.batch_size = batch_size
        self.n_components_pca = n_components_pca
        self.random_state = random_state
        self.pca = IncrementalPCA(n_components=n_components_pca, batch_size=batch_size)
        self.reducer = umap.UMAP(random_state=random_state)
        
    def load_embeddings_efficiently(self, file_path, display_total = False):
        """Load embeddings in batches to avoid memory issues"""
        print('Loading embeddings efficiently...')
        
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
        
        return embeddings, feat_dim, total_embeddings, labels
    
    def fit_incremental_pca(self, embeddings_data, feat_dim):
        """Fit PCA incrementally to handle large datasets"""
        print('Fitting Incremental PCA...')
        
        batch = []
        batch_count = 0
        
        for embedding_group in tqdm(embeddings_data):
            if isinstance(embedding_group, list):
                batch.extend(embedding_group)
            else:
                batch.append(embedding_group)
            
            # Process batch when it reaches batch_size
            if len(batch) >= self.batch_size:
                batch_array = np.array(batch)
                self.pca.partial_fit(batch_array)
                batch = []
                batch_count += 1
                
                # Clean up memory
                del batch_array
                gc.collect()
        
        # Process remaining batch
        if batch:
            batch_array = np.array(batch)
            self.pca.partial_fit(batch_array)
            del batch_array
            gc.collect()
    
    def transform_embeddings(self, embeddings_data, feat_dim):
        """Transform embeddings using fitted PCA"""
        print('Transforming embeddings with PCA...')
        
        transformed_embeddings = []
        batch = []
        
        for embedding_group in tqdm(embeddings_data):
            if isinstance(embedding_group, list):
                batch.extend(embedding_group)
            else:
                batch.append(embedding_group)
            
            # Process batch when it reaches batch_size
            if len(batch) >= self.batch_size:
                batch_array = np.array(batch)
                transformed_batch = self.pca.transform(batch_array)
                transformed_embeddings.append(transformed_batch)
                batch = []
                
                # Clean up memory
                del batch_array
                gc.collect()
        
        # Process remaining batch
        if batch:
            batch_array = np.array(batch)
            transformed_batch = self.pca.transform(batch_array)
            transformed_embeddings.append(transformed_batch)
            del batch_array
            gc.collect()
        
        # Concatenate all transformed batches
        return np.vstack(transformed_embeddings)
    
    def fit_umap_on_subset_transform_all(self, embeddings, fit_sample_size=50000):
        """Fit UMAP on a subset but transform all embeddings"""
        if len(embeddings) > fit_sample_size:
            print(f'Fitting UMAP on {fit_sample_size} samples, then transforming all {len(embeddings)}...')
            
            # Fit on subset
            fit_indices = np.random.choice(len(embeddings), fit_sample_size, replace=False)
            fit_subset = embeddings[fit_indices]
            self.reducer.fit(fit_subset)
            
            # Transform all embeddings in batches
            batch_size = 10000
            all_transformed = []
            
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Transforming batches"):
                batch = embeddings[i:i+batch_size]
                batch_transformed = self.reducer.transform(batch)
                all_transformed.append(batch_transformed)
                
                # Clean up memory
                del batch, batch_transformed
                gc.collect()
            
            return np.vstack(all_transformed)
        else:
            # If small enough, fit and transform normally
            return self.reducer.fit_transform(embeddings)
    
    def process_all_embeddings(self, file_path, fit_sample_size=50000):
        """Process ALL embeddings - fit on subset, transform all"""
        
        # Load embeddings efficiently
        embeddings_data, feat_dim, total_embeddings, labels = self.load_embeddings_efficiently(file_path)
        
        # Fit PCA incrementally
        self.fit_incremental_pca(embeddings_data, feat_dim)
        
        # Transform ALL embeddings
        pca_embeddings = self.transform_embeddings(embeddings_data, feat_dim)
        
        # Clean up original data
        del embeddings_data
        gc.collect()
        
        print(f'PCA completed. Shape: {pca_embeddings.shape}')
        
        # Fit UMAP on subset but transform ALL
        umap_embeddings = self.fit_umap_on_subset_transform_all(pca_embeddings, fit_sample_size)
        
        return pca_embeddings, umap_embeddings, labels
    
    def process_all_with_approximation(self, file_path):
        """Alternative: Use parametric UMAP to handle all embeddings"""
        try:
            # Parametric UMAP can handle larger datasets
            import umap.parametric_umap as parametric_umap
            
            embeddings_data, feat_dim, total_embeddings = self.load_embeddings_efficiently(file_path)
            self.fit_incremental_pca(embeddings_data, feat_dim)
            pca_embeddings = self.transform_embeddings(embeddings_data, feat_dim)
            
            # Use parametric UMAP (can handle all points)
            parametric_reducer = parametric_umap.ParametricUMAP(
                random_state=self.random_state,
                verbose=True
            )
            
            print('Computing Parametric UMAP on all embeddings...')
            umap_embeddings = parametric_reducer.fit_transform(pca_embeddings)
            
            return pca_embeddings, umap_embeddings
            
        except ImportError:
            print("Parametric UMAP not available. Use: pip install umap-learn[parametric_umap]")
            return None, None

# Alternative approach using random projection (faster than PCA)
class FastEmbeddingsProcessor:
    def __init__(self, n_components=100, random_state=42):
        self.projector = GaussianRandomProjection(n_components=n_components, random_state=random_state)
        self.reducer = umap.UMAP(random_state=random_state)
    
    def process_all_embeddings_fast(self, file_path, fit_sample_size=50000):
        """Process ALL embeddings using random projection"""
        print('Loading embeddings...')
        
        _embeddings = torch.load(file_path)
        embeddings = []
        for e in _embeddings:
            embeddings.extend(e)
        
        feat_dim = embeddings[0].shape[-1]
        embeddings = np.array(embeddings).reshape(-1, feat_dim)
        
        print(f'Loaded {len(embeddings)} embeddings with {feat_dim} features')
        
        # Random projection on ALL embeddings
        print('Computing Random Projection on all embeddings...')
        projected_embeddings = self.projector.fit_transform(embeddings)
        
        # Clean up
        del embeddings
        gc.collect()
        
        # Fit UMAP on subset, transform all
        if len(projected_embeddings) > fit_sample_size:
            print(f'Fitting UMAP on {fit_sample_size} samples, transforming all...')
            
            # Fit on subset
            fit_indices = np.random.choice(len(projected_embeddings), fit_sample_size, replace=False)
            fit_subset = projected_embeddings[fit_indices]
            self.reducer.fit(fit_subset)
            
            # Transform all in batches
            batch_size = 10000
            all_umap = []
            
            for i in tqdm(range(0, len(projected_embeddings), batch_size), desc="UMAP transform"):
                batch = projected_embeddings[i:i+batch_size]
                batch_umap = self.reducer.transform(batch)
                all_umap.append(batch_umap)
                
                del batch, batch_umap
                gc.collect()
            
            umap_result = np.vstack(all_umap)
        else:
            umap_result = self.reducer.fit_transform(projected_embeddings)
        
        return projected_embeddings, umap_result

# Usage examples
def main():
    file_path = '/home/tomwelch/Cambridge/Embeddings/small_dataset_embs_518.pt'
    
    # Method 1: Keep ALL embeddings with incremental PCA + fit/transform UMAP
    processor = LargeEmbeddingsProcessor(batch_size=5000, n_components_pca=100)
    pca_embeddings, umap_all, labels = processor.process_all_embeddings(
        file_path, fit_sample_size=50000
    )
    
    print(f"Final shapes - PCA: {pca_embeddings.shape}, UMAP: {umap_all.shape}")
        
    plt.figure(figsize=(20, 20), dpi=200)

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
    
    # Method 2: Parametric UMAP (if available) - can handle all points natively
    # pca_embeddings_param, umap_param = processor.process_all_with_approximation(file_path)
    # if umap_param is not None:
    #     plt.figure(figsize=(12, 8))
    #     plt.scatter(umap_param[:, 0], umap_param[:, 1], alpha=0.6, s=0.5)
    #     plt.title(f'Parametric UMAP - ALL {len(umap_param)} embeddings')
    #     plt.show()
    
    # Method 3: Fast random projection approach keeping all
    # fast_processor = FastEmbeddingsProcessor(n_components=100)
    # proj_embeddings, umap_result_all = fast_processor.process_all_embeddings_fast(
    #     file_path, fit_sample_size=50000
    # )
    
    # plt.figure(figsize=(12, 8))
    # plt.scatter(umap_result_all[:, 0], umap_result_all[:, 1], alpha=0.6, s=0.5)
    # plt.title(f'Random Projection + UMAP - ALL {len(umap_result_all)} embeddings')
    # plt.show()

if __name__ == "__main__":
    main()