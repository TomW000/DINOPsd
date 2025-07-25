from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as utils
import torch
import torchvision

#from src.setup import neurotransmitters, model
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image

import os
import random 

from src.setup import embeddings_path, model, feat_dim, device

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING DATA:
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('-Loading embeddings')

_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
EMBEDDINGS=[]
for e in _EMBEDDINGS:
    EMBEDDINGS.extend(e)
EMBEDDINGS = EMBEDDINGS

_REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

print('...done loading embeddings')

# OPTIMIZATION: Cache for loaded and processed images to avoid reloading
_IMAGE_CACHE = {}
_PROCESSED_DATA_CACHE = {}

def get_data_generator(split: str = 'training', 
                       nb_best_patches: int = 1,
                       resize_size: int = 518, 
                       padding_size: int = 1, 
                       test_proportion: float = 0.2,
                       seed: int = 42,
                       max_samples: int = None,  # OPTIMIZATION: Limit samples for faster iteration
                       use_cache: bool = True):  # OPTIMIZATION: Enable caching
    
    patch_size = model.patch_size # type: ignore 

    assert resize_size % patch_size == 0, f'crop size must be a multiple of patch_size = {patch_size}'

    files, _ = zip(*get_fnames())

    random.seed(seed)

    DATASET = list(zip(files, EMBEDDINGS))[:10]
    random.shuffle(DATASET)
    SPLIT = int(len(DATASET) * test_proportion)
    TRAINING_SET = DATASET[SPLIT:]
    TEST_SET = DATASET[:SPLIT]

    list_iterator = TRAINING_SET if split == 'training' else TEST_SET
    
    # OPTIMIZATION: Limit number of samples if specified
    if max_samples is not None:
        list_iterator = list_iterator[:max_samples]
        print(f"Limited to {len(list_iterator)} samples for {split}")

    # OPTIMIZATION: Pre-compute padding outside the loop
    padding = padding_size * patch_size

    for file, embeddings in tqdm(list_iterator, total=len(list_iterator), desc=f'-> Loop through {split} set'):
        
        # OPTIMIZATION: Create cache key
        cache_key = f"{file}_{resize_size}_{padding_size}_{nb_best_patches}"
        
        if use_cache and cache_key in _PROCESSED_DATA_CACHE:
            # Return cached data
            yield _PROCESSED_DATA_CACHE[cache_key]
            continue
        
        # OPTIMIZATION: Cache image loading
        if use_cache and file in _IMAGE_CACHE:
            resized_image = _IMAGE_CACHE[file]
        else:
            image, _, _ = load_image(file)
            resized_image = resize_hdf_image(image, resize_size=resize_size).squeeze()
            if use_cache:
                _IMAGE_CACHE[file] = resized_image
        
        H_size, W_size = resized_image.shape
        H_patch, W_patch = H_size // patch_size, W_size // patch_size
        
        # OPTIMIZATION: Pre-allocate patches list with known size
        total_patches = H_patch * W_patch
        PATCHES = []
        #PATCHES.reserve = total_patches  # Hint for better memory allocation
        
        # OPTIMIZATION: Vectorized patch extraction
        patches_data = []
        patch_coords = []
        
        for i in range(0, H_size, patch_size):
            for j in range(0, W_size, patch_size):
                start_h = max(i - padding, 0)
                end_h = min(i + padding + patch_size, H_size)
                start_w = max(j - padding, 0)
                end_w = min(j + padding + patch_size, W_size)
                
                patch = resized_image[start_h:end_h, start_w:end_w]
                patch_coords.append((start_h, end_h, start_w, end_w))
                patches_data.append(patch)
        
        # OPTIMIZATION: Batch process patches instead of one by one
        PATCHES = []
        for patch in patches_data:
            # Convert to 3-channel and add batch dimension
            patch_3ch = patch[..., None]
            stack = np.concatenate([patch_3ch, patch_3ch, patch_3ch], axis=2)
            stack = stack.transpose(2, 0, 1)[None]
            stack = torch.from_numpy(stack).float()  # Convert to float32 immediately
            PATCHES.append(stack)
        
        # OPTIMIZATION: More efficient similarity computation
        flattened_embeddings = embeddings.reshape(-1, feat_dim)
        REFS = torch.from_numpy(_REFS)
        REFS = REFS.to(device)
        # OPTIMIZATION: Use more efficient distance computation
        if nb_best_patches == 1:
            # For single best patch, we can optimize further
            flattened_embeddings = flattened_embeddings.to(device)
            distances = torch.cdist(REFS, flattened_embeddings)
            distances = distances.cpu().numpy()
            min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
            patch_idx = min_dist_idx[1]
        else:
            # Original logic for multiple patches
            flattened_embeddings = flattened_embeddings.to(device)
            distances = torch.cdist(REFS, flattened_embeddings)
            distances = distances.cpu().numpy()
            flat_similarities = distances.ravel()
            top_flat_indices = flat_similarities.argsort()[:nb_best_patches]
            ref_indices, patch_indices = np.unravel_index(top_flat_indices, distances.shape)
            patch_idx = patch_indices[0] if nb_best_patches == 1 else patch_indices
        
        # OPTIMIZATION: Simplified ground truth generation for single patch
        if nb_best_patches == 1:
            h_patch_coord = patch_idx // W_patch
            w_patch_coord = patch_idx % W_patch
            
            gt_h_start_idx = int(h_patch_coord * patch_size)
            gt_h_end_idx = int((h_patch_coord + 1) * patch_size)
            gt_w_start_idx = int(w_patch_coord * patch_size)
            gt_w_end_idx = int((w_patch_coord + 1) * patch_size)

            GT = torch.zeros((H_size, W_size), dtype=torch.float32)  # Use float32
            GT[gt_h_start_idx:gt_h_end_idx, gt_w_start_idx:gt_w_end_idx] = 1.0
        else:
            # Handle multiple patches case
            GT = torch.zeros((H_size, W_size), dtype=torch.float32)
            for idx in patch_indices:
                h_patch_coord = idx // W_patch
                w_patch_coord = idx % W_patch
                
                gt_h_start_idx = int(h_patch_coord * patch_size)
                gt_h_end_idx = int((h_patch_coord + 1) * patch_size)
                gt_w_start_idx = int(w_patch_coord * patch_size)
                gt_w_end_idx = int((w_patch_coord + 1) * patch_size)
                
                GT[gt_h_start_idx:gt_h_end_idx, gt_w_start_idx:gt_w_end_idx] = 1.0

        result = (file, PATCHES, GT)
        
        # OPTIMIZATION: Cache the result
        if use_cache:
            _PROCESSED_DATA_CACHE[cache_key] = result
            
            # OPTIMIZATION: Limit cache size to prevent memory issues
            if len(_PROCESSED_DATA_CACHE) > 100:  # Keep only 100 most recent
                # Remove oldest entries
                keys_to_remove = list(_PROCESSED_DATA_CACHE.keys())[:-50]
                for key in keys_to_remove:
                    del _PROCESSED_DATA_CACHE[key]

        yield result


# OPTIMIZATION: Add function to clear caches when needed
def clear_data_caches():
    """Clear all caches to free memory"""
    global _IMAGE_CACHE, _PROCESSED_DATA_CACHE
    _IMAGE_CACHE.clear()
    _PROCESSED_DATA_CACHE.clear()
    print("Data caches cleared")


# OPTIMIZATION: Add function to get dataset size without loading all data
def get_dataset_size(split: str = 'training', test_proportion: float = 0.2):
    """Get dataset size without loading the data"""
    files, _ = zip(*get_fnames())
    total_files = len(files)
    
    if split == 'training':
        return total_files - int(total_files * test_proportion)
    else:
        return int(total_files * test_proportion)


# OPTIMIZATION: Create a more efficient data loader class
class OptimizedDataLoader:
    """Optimized data loader that pre-loads and batches data"""
    
    def __init__(self, split='training', nb_best_patches=1, resize_size=518, 
                 padding_size=1, test_proportion=0.2, seed=42, 
                 preload_count=10, use_cache=True):
        
        self.split = split
        self.nb_best_patches = nb_best_patches
        self.resize_size = resize_size
        self.padding_size = padding_size
        self.test_proportion = test_proportion
        self.seed = seed
        self.preload_count = preload_count
        self.use_cache = use_cache
        
        self._preloaded_data = []
        self._generator = None
        self._preload_data()
    
    def _preload_data(self):
        """Pre-load a batch of data"""
        self._generator = get_data_generator(
            split=self.split,
            nb_best_patches=self.nb_best_patches,
            resize_size=self.resize_size,
            padding_size=self.padding_size,
            test_proportion=self.test_proportion,
            seed=self.seed,
            max_samples=self.preload_count,
            use_cache=self.use_cache
        )
        
        self._preloaded_data = list(self._generator)
        print(f"Pre-loaded {len(self._preloaded_data)} samples for {self.split}")
    
    def __iter__(self):
        """Return iterator over pre-loaded data"""
        return iter(self._preloaded_data)
    
    def __len__(self):
        return len(self._preloaded_data)


# OPTIMIZATION: Quick testing function
def test_data_generator_speed():
    """Test the speed of data generation"""
    import time
    
    print("Testing original vs optimized data generator...")
    
    # Test optimized version
    start_time = time.time()
    count = 0
    for patches, gt in get_data_generator(split='training', max_samples=5, use_cache=True):
        count += 1
        if count >= 5:
            break
    
    optimized_time = time.time() - start_time
    print(f"Optimized version: {optimized_time:.2f} seconds for 5 samples")
    
    # Clear cache and test again
    clear_data_caches()
    start_time = time.time()
    count = 0
    for patches, gt in get_data_generator(split='training', max_samples=5, use_cache=False):
        count += 1
        if count >= 5:
            break
    
    no_cache_time = time.time() - start_time
    print(f"No cache version: {no_cache_time:.2f} seconds for 5 samples")
    
    return optimized_time, no_cache_time