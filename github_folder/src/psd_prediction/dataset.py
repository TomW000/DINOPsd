from sklearn.model_selection import KFold
import torch
import numpy as np
import os
import random 
from torch.utils.data import Dataset
import torch.utils.data as utils
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
# Using torch.cdist instead of sklearn euclidean_distances for better performance
from timeit import default_timer as timer
import gc

from src.DinoPsd import DinoPsd_pipeline
from src.DinoPsd_utils import get_img_processing_f

from src.setup import embeddings_path, feat_dim, device, model, resize_size

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING DATA:
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#device = 'cpu' # TODO: Change to GPU

print('-Loading embeddings')

dataset_size = 'small'
dataset = dataset_size + '_dataset_embs_518.pt'
ref_dataset = dataset_size + '_mean_ref_518_Aug=False_k=10.pt'

EMBEDDINGS = torch.load(os.path.join(embeddings_path, dataset))

if dataset_size == 'small':
    l = []
    for e in EMBEDDINGS:
        l.extend(e)
    EMBEDDINGS = torch.from_numpy(np.array(l)).to(device)
    
    print(f'...loaded embeddings shape: {EMBEDDINGS.shape}')

REFS = torch.load(os.path.join(embeddings_path, ref_dataset), weights_only=False)

print('...done loading embeddings')

# Convert to tensors and move to device for efficient processing
REFS = torch.tensor(REFS, device=device) if isinstance(REFS, np.ndarray) else torch.from_numpy(np.array(REFS)).to(device)
mean_psd_embedding = torch.mean(REFS, dim=0)

PSD_list, REST_list = [], []

nb_best_patches = 1



already_flattened = False

if already_flattened:
    patch_size = model.patch_size
    batch_size = (resize_size // patch_size) ** 2
else:
    batch_size = 600




def find_top_k_patches_efficient(refs, image, k=1):
    """
    Efficiently find top k patches using torch.cdist and partial sorting
    """
        
    # Ensure tensors are on the same device
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).to(refs.device).to(refs.dtype)
    else:
        image = image.to(refs.device).to(refs.dtype)
    
    if image.ndim > 2:
        flattened_image = image.reshape(-1, feat_dim)
    elif image.ndim == 2:
        flattened_image = image
    else:
        raise ValueError("Input must be a 2D or 3D tensor.")
    
    # Compute distances using torch.cdist (more efficient than sklearn)
    distances = torch.cdist(refs, flattened_image)  # Shape: (n_refs, n_patches)
    
    # Find top k using topk (more efficient than full sort)
    flat_distances = distances.view(-1)
    
    if k == 1:
        # For k=1, just find the minimum
        top_flat_idx = torch.argmin(flat_distances)
        top_flat_indices = [top_flat_idx.item()]
    else:
        # Use topk for k > 1 (more efficient than argsort)
        _, top_flat_indices = torch.topk(flat_distances, k, largest=False)
        top_flat_indices = top_flat_indices.cpu().numpy()
    
    # Map back to (ref_idx, patch_idx)
    ref_indices, patch_indices = np.unravel_index(top_flat_indices, distances.shape)
    
    return ref_indices, patch_indices

print('-Processing batches and finding PSD/REST patches')

for i in tqdm(range(0, len(EMBEDDINGS), batch_size), desc='Looping through batches'):
    end_idx = min(len(EMBEDDINGS), i + batch_size)
    batch = EMBEDDINGS[i:end_idx]
    
    # Convert batch to tensor if needed and move to device
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    else:
        batch = torch.from_numpy(np.array(batch)).to(device)

    if already_flattened:
        flattened_image = batch
        
        # Use efficient top-k finding with torch
        ref_indices, patch_indices = find_top_k_patches_efficient(REFS, flattened_image, nb_best_patches)

        # Save top patches (convert to CPU for storage)
        selected_patches = flattened_image[patch_indices].cpu()
        PSD_list.extend(selected_patches)

        # Create mask for rest patches more efficiently
        mask = torch.ones(len(flattened_image), dtype=torch.bool, device=device)
        unique_patch_indices = torch.tensor(np.unique(patch_indices), device=device)
        mask[unique_patch_indices] = False
        
        # Add rest patches (convert to CPU for storage)
        rest_patches = flattened_image[mask].cpu()
        REST_list.extend(rest_patches)

    else:
        for image in tqdm(batch, desc='-> Comparing embeddings to reference'):
            H_patch, W_patch, _ = image.shape  # patch grid size
            flattened_image = image.reshape(-1, feat_dim)
            
            # Use efficient top-k finding with torch
            ref_indices, patch_indices = find_top_k_patches_efficient(REFS, flattened_image, nb_best_patches)

            # Save top patches (convert to CPU for storage)
            selected_patches = flattened_image[patch_indices].cpu()
            PSD_list.extend(selected_patches)

            # Create mask for rest patches
            mask = torch.ones(flattened_image.shape[0], dtype=torch.bool, device=device)
            unique_patch_indices = torch.tensor(np.unique(patch_indices), device=device)
            mask[unique_patch_indices] = False
            
            # Add rest patches (convert to CPU for storage)
            rest_patches = flattened_image[mask].cpu()
            REST_list.extend(rest_patches)

    # Clean up batch from memory
    del batch
    if i % 1 == 0:  # Periodic garbage collection
        gc.collect()

# Convert PSD to tensor (already on CPU from processing)
PSD = torch.stack(PSD_list) if PSD_list else torch.empty(0, feat_dim)
print(f'-PSD collected: {PSD.shape}')

# Clear PSD_list from memory
del PSD_list
gc.collect()

print('-Sorting rest embeddings...(this may take a while - check memory usage)')
start = timer()

# More memory-efficient sorting approach
def sort_embeddings_by_distance_efficient(embeddings_list, reference_embedding, chunk_size=50000):
    """
    Sort embeddings by distance to reference using torch.cdist for better performance
    """
    total_embeddings = len(embeddings_list)
    
    if total_embeddings <= chunk_size:
        # Small enough to process at once
        embeddings_tensor = torch.stack(embeddings_list).to(reference_embedding.device)
        distances = torch.cdist(reference_embedding.unsqueeze(0), embeddings_tensor).squeeze(0)
        sorted_indices = torch.argsort(distances)
        return embeddings_tensor[sorted_indices].cpu()
    
    # Process in chunks
    print(f'Processing {total_embeddings} embeddings in chunks of {chunk_size}')
    
    # First pass: compute distances and create index pairs
    distance_index_pairs = []
    
    for chunk_start in tqdm(range(0, total_embeddings, chunk_size), desc='Computing distances'):
        chunk_end = min(chunk_start + chunk_size, total_embeddings)
        chunk_embeddings = torch.stack(embeddings_list[chunk_start:chunk_end]).to(reference_embedding.device)
        
        chunk_distances = torch.cdist(reference_embedding.unsqueeze(0), chunk_embeddings).squeeze(0)
        
        # Store (distance, global_index) pairs
        for local_idx, distance in enumerate(chunk_distances):
            global_idx = chunk_start + local_idx
            distance_index_pairs.append((distance.item(), global_idx))
        
        del chunk_embeddings, chunk_distances
        torch.cuda.empty_cache() if reference_embedding.is_cuda else gc.collect()
    
    # Sort by distance
    print('Sorting by distances...')
    distance_index_pairs.sort(key=lambda x: x[0])
    
    # Second pass: reconstruct sorted array
    print('Reconstructing sorted array...')
    sorted_embeddings = []
    
    for distance, global_idx in tqdm(distance_index_pairs, desc='Reconstructing'):
        sorted_embeddings.append(embeddings_list[global_idx])
    
    return torch.stack(sorted_embeddings)

# Apply efficient sorting with torch
REST_tensor = sort_embeddings_by_distance_efficient(REST_list, mean_psd_embedding)
REST = REST_tensor

# Clean up intermediate variables
del REST_list, REST_tensor
torch.cuda.empty_cache() if device == 'cuda' else gc.collect()

end = timer()
print(f'...done sorting rest embeddings - it took {end-start:.2f} seconds')

# Create labels
PSD_LABELS = torch.ones(PSD.shape[0])
REST_LABELS = torch.zeros(REST.shape[0])

# Create labeled datasets
LABELLED_PSD = list(zip(PSD, PSD_LABELS))
LABELLED_REST = list(zip(REST, REST_LABELS))

print(f'-PSD shape: {PSD.shape}, Rest shape: {REST.shape}')

# Final memory cleanup
torch.cuda.empty_cache() if device == 'cuda' else gc.collect()
print('-Processing complete')

print('Saving datasets ...')
torch.save(LABELLED_PSD, os.path.join(embeddings_path, 'LABELLED_PSD.pt'))
torch.save(LABELLED_REST, os.path.join(embeddings_path, 'LABELLED_REST.pt'))
print('... done saving datasets')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEFINING CUSTOM DATASETS AND DATASET GENERATORS:
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
class Custom_Dataset(Dataset):
    def __init__(self,
                 LABELLED_PSD: torch.Tensor = LABELLED_PSD,
                 LABELLED_REST: torch.Tensor = LABELLED_REST,
                 sliding_window: bool = None,
                 stride: int = 100,
                 set_type: str = 'training',
                 test_proportion: float = 0.2,
                 n: int = 0,
                 dataset_bias: int = 1, 
                 seed: int = 42):
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Validate inputs
        assert set_type in {'training', 'test'}, 'set_type must be either training or test'
        assert 0 < test_proportion < 1, 'test_proportion must be between 0 and 1'
        assert dataset_bias > 0, 'dataset_bias must be positive'
        assert stride > 0, 'stride must be positive'
        
        if sliding_window and LABELLED_REST is None:
            raise ValueError("LABELLED_REST must be provided when sliding_window=True")
        
        self.set_type = set_type
        self.test_proportion = test_proportion
        self.sliding_window = sliding_window
        self.stride = stride
        self.n = n

        # Process PSD data 
        self.psd = LABELLED_PSD
        self.len_psd = len(LABELLED_PSD) // dataset_bias
        
        # Select REST data based on mode
        if sliding_window:
            window_size = self.len_psd
            start_idx = n * stride
            end_idx = min(n * stride + window_size, len(LABELLED_REST))
            
            # Validate window bounds
            if start_idx >= len(LABELLED_REST):
                raise IndexError(f"Window start index {start_idx} exceeds LABELLED_REST length {len(LABELLED_REST)}")
            
            self.rest = LABELLED_REST[start_idx:end_idx]
        elif sliding_window is False:   
            start_idx = self.len_psd * n
            end_idx = min(self.len_psd * (n+1), len(LABELLED_REST))
            
            # Validate fold bounds
            if start_idx >= len(LABELLED_REST):
                raise IndexError(f"Fold start index {start_idx} exceeds LABELLED_REST length {len(LABELLED_REST)}")
            
            self.rest = LABELLED_REST[start_idx:end_idx]
            
        else:
            self.rest = LABELLED_REST

        # Combine and shuffle dataset
        self.DATASET = self.psd + self.rest
        random.shuffle(self.DATASET)
        
        # Split into training and test sets
        self.SPLIT = int(len(self.DATASET) * self.test_proportion)
        self.TRAINING_SET = self.DATASET[self.SPLIT:]
        self.TEST_SET = self.DATASET[:self.SPLIT]

        # Select appropriate dataset
        self.data = self.TRAINING_SET if set_type == 'training' else self.TEST_SET

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
    
    def get_class_distribution(self) -> dict[int, int]:
        """Return the distribution of classes in the current dataset"""
        labels = [item[1].item() if isinstance(item[1], torch.Tensor) else item[1] for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.astype(int), counts))
    
    def get_dataset_info(self) -> dict[str, any]:
        """Return comprehensive information about the dataset"""
        class_dist = self.get_class_distribution()
        return {
            'set_type': self.set_type,
            'total_samples': len(self.data),
            'psd_samples': len(self.psd),
            'rest_samples': len(self.rest),
            'class_distribution': class_dist,
            'test_proportion': self.test_proportion,
            'sliding_window': self.sliding_window,
            'window_number': self.n,
            'stride': self.stride if self.sliding_window else None
        }


    def get_balance_ratio(self) -> float:
        """Return the ratio of positive to negative samples"""
        class_dist = self.get_class_distribution()
        pos_count = class_dist.get(1, 0)
        neg_count = class_dist.get(0, 0)
        return pos_count / neg_count if neg_count > 0 else float('inf')


def dataset_generator(sliding_window: bool=False,
                      LABELLED_PSD_SET: list=LABELLED_PSD,
                      LABELLED_REST_SET: list=LABELLED_REST, 
                      test_proportion: float=0.2, 
                      stride: int=100, 
                      train_batch_size: int=50, 
                      test_batch_size: int=50,
                      seed: int = 42):
    """
    A generator of PSD prediction datasets. Either uses a sliding window or non-overlapping folds.

    Args:
        sliding_window: If True, use a sliding window to generate datasets. Otherwise, use non-overlapping folds.
        LABELLED_PSD_SET: The list of labelled PSD data.
        LABELLED_REST_SET: The list of labelled REST data.
        test_proportion: The proportion of data to be used for testing.
        stride: The stride of the sliding window.
        train_batch_size: The batch size of the training data loader.
        test_batch_size: The batch size of the test data loader.
        seed: The random seed to use for reproducibility.

    Yields:
        A tuple of (training data loader, test data loader, dataset info) for each iteration of the generator.
    """

    if sliding_window:
        window_size = len(LABELLED_PSD_SET)
        max_windows = (len(LABELLED_REST_SET) - window_size) // stride + 1
        
        for k in range(max_windows):
            try:
                training_dataset = Custom_Dataset(
                    LABELLED_PSD=LABELLED_PSD_SET,
                    LABELLED_REST=LABELLED_REST_SET,
                    sliding_window=True,
                    stride=stride,
                    set_type='training',
                    test_proportion=test_proportion,
                    n=k,
                    dataset_bias=1,
                    seed=seed
                )
                
                test_dataset = Custom_Dataset(
                    LABELLED_PSD=LABELLED_PSD_SET,
                    LABELLED_REST=LABELLED_REST_SET,
                    sliding_window=True,
                    stride=stride,
                    set_type='test',
                    test_proportion=test_proportion,
                    n=k,
                    dataset_bias=1,
                    seed=seed
                )
                
            except IndexError as e:
                print(f"Skipping window {k}: {e}")
                continue
                
            train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
            
            yield train_loader, test_loader, training_dataset.get_dataset_info()

        
    else:
        n_splits =  len(LABELLED_REST_SET) // len(LABELLED_PSD_SET)

        for k in range(n_splits):
            try:
                training_dataset = Custom_Dataset(
                    LABELLED_PSD=LABELLED_PSD_SET,
                    LABELLED_REST=LABELLED_REST_SET,
                    sliding_window=False,
                    set_type='training',
                    test_proportion=test_proportion,
                    n=k,
                    dataset_bias=1,
                    seed=seed
                )
                
                test_dataset = Custom_Dataset(
                    LABELLED_PSD=LABELLED_PSD_SET,
                    LABELLED_REST=LABELLED_REST_SET,
                    sliding_window=False,
                    set_type='test',
                    test_proportion=test_proportion,
                    n=k,
                    dataset_bias=1,
                    seed=seed
                )
                
            except IndexError as e:
                print(f"Skipping window {k}: {e}")
                continue

            train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
                
            yield train_loader, test_loader, training_dataset.get_dataset_info()


# Example usage and testing
if __name__ == "__main__":
    # Test cross-validation
    print("Cross-validation datasets:")
    for i, (train_loader, test_loader, info) in enumerate(cross_validation_datasets_generator(0.2)):
        print(f"Fold {i}: {info}")
        if i >= 2:  # Just show first 3 folds
            break
    
    # Test sliding window (assuming FILTERED_REST_SET is defined)
    # print("\nSliding window datasets:")
    # for i, (train_loader, test_loader, info) in enumerate(sliding_window_datasets_generator(
    #     FILTERED_REST_SET, test_proportion=0.2, stride=100)):
    #     print(f"Window {i}: {info}")
    #     if i >= 2:  # Just show first 3 windows
    #         break
'''
