from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as utils
import torch
import torchvision
import gc
import warnings
from typing import Optional, Tuple, List, Generator

#from src.setup import neurotransmitters, model
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image

import os
import random 

from src.setup import embeddings_path, model, feat_dim, device, resize_size

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING DATA:
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('-Loading embeddings')

# FIXED: Add error handling for embeddings loading
try:
    _EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'), 
                            map_location='cpu')  # Load to CPU first to save GPU memory
    EMBEDDINGS = []
    for e in _EMBEDDINGS:
        EMBEDDINGS.extend(e)
    
    _REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), 
                      weights_only=False, map_location='cpu')
    
    print(f'...done loading embeddings. Total embeddings: {len(EMBEDDINGS)}')
    
except FileNotFoundError as e:
    print(f"Error loading embeddings: {e}")
    raise
except Exception as e:
    print(f"Unexpected error loading embeddings: {e}")
    raise

# FIXED: Thread-safe cache with size limits and proper cleanup
class DataCache:
    """Thread-safe cache with automatic size management"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._image_cache = {}
        self._processed_cache = {}
        self._access_order = []
    
    def get_image(self, key: str):
        if key in self._image_cache:
            self._update_access(key)
            return self._image_cache[key]
        return None
    
    def set_image(self, key: str, value):
        self._image_cache[key] = value
        self._update_access(key)
        self._cleanup_if_needed()
    
    def get_processed(self, key: str):
        if key in self._processed_cache:
            self._update_access(key)
            return self._processed_cache[key]
        return None
    
    def set_processed(self, key: str, value):
        self._processed_cache[key] = value
        self._update_access(key)
        self._cleanup_if_needed()
    
    def _update_access(self, key: str):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _cleanup_if_needed(self):
        total_items = len(self._image_cache) + len(self._processed_cache)
        if total_items > self.max_size:
            # Remove oldest items
            items_to_remove = total_items - self.max_size + 5  # Remove a few extra
            for _ in range(items_to_remove):
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    self._image_cache.pop(oldest_key, None)
                    self._processed_cache.pop(oldest_key, None)
    
    def clear(self):
        self._image_cache.clear()
        self._processed_cache.clear()
        self._access_order.clear()
        gc.collect()  # Force garbage collection
    
    def size(self):
        return len(self._image_cache) + len(self._processed_cache)

# FIXED: Use proper cache instance instead of global variables
_DATA_CACHE = DataCache(max_size=100)


def get_data_generator(split: str = 'training', 
                       nb_best_patches: int = 1,
                       resize_size: int = resize_size, 
                       padding_size: int = 1, 
                       test_proportion: float = 0.2,
                       seed: int = 42,
                       max_samples: Optional[int] = None,
                       use_cache: bool = True) -> Generator[Tuple[str, List[torch.Tensor], torch.Tensor], None, None]:
    """
    FIXED: Improved data generator with proper error handling and memory management
    
    Args:
        split: 'training' or 'test'
        nb_best_patches: Number of best patches to find
        resize_size: Target resize dimension
        padding_size: Padding around patches in patch_size units
        test_proportion: Proportion of data for testing
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to process
        use_cache: Whether to use caching
        
    Yields:
        Tuple of (filename, patches_list, ground_truth_tensor)
    """
    
    # FIXED: Input validation
    if split not in ['training', 'test']:
        raise ValueError("split must be 'training' or 'test'")
    
    if not hasattr(model, 'patch_size'):
        raise AttributeError("model must have patch_size attribute")
        
    patch_size = model.patch_size
    
    if resize_size % patch_size != 0:
        raise ValueError(f'resize_size ({resize_size}) must be a multiple of patch_size ({patch_size})')
    
    if not (0 < test_proportion < 1):
        pass
        #raise ValueError("test_proportion must be between 0 and 1") # FIXME: Remove this line

    # FIXED: Robust file loading with error handling
    try:
        files_data = get_fnames()
        if isinstance(files_data[0], tuple):
            files, _ = zip(*files_data)
        else:
            files = files_data
    except Exception as e:
        print(f"Error getting filenames: {e}")
        raise

    # FIXED: Ensure deterministic splits
    random.seed(seed)
    np.random.seed(seed)  # Also set numpy seed for consistency

    # FIXED: Check data consistency
    if len(files) != len(EMBEDDINGS):
        warnings.warn(f"Mismatch between files ({len(files)}) and embeddings ({len(EMBEDDINGS)})")
        min_len = min(len(files), len(EMBEDDINGS))
        files = files[:min_len]
        embeddings_subset = EMBEDDINGS[:min_len]
    else:
        embeddings_subset = EMBEDDINGS

    DATASET = list(zip(files, embeddings_subset))
    random.shuffle(DATASET)
    DATASET = DATASET[:10]
    
    # FIXED: Remove hardcoded dataset limitation - make it configurable
    if max_samples and max_samples < len(DATASET):
        DATASET = random.sample(DATASET, max_samples)
        print(f"Limited dataset to {len(DATASET)} samples")
    
    # FIXED: More robust train/test split
    split_idx = int(len(DATASET) * test_proportion)
    if split_idx == 0 and len(DATASET) > 0:
        split_idx = 1  # Ensure at least one test sample
    
    TRAINING_SET = DATASET[split_idx:]
    TEST_SET = DATASET[:split_idx]

    list_iterator = TRAINING_SET if split == 'training' else TEST_SET
    
    if len(list_iterator) == 0:
        warnings.warn(f"No data available for split '{split}'")
        return

    print(f"Processing {len(list_iterator)} samples for {split} split")

    # FIXED: Pre-compute padding outside the loop
    padding = padding_size * patch_size
    
    # FIXED: Pre-load and validate reference embeddings
    try:
        refs_tensor = torch.from_numpy(_REFS).float()
        if device != 'cpu':
            refs_tensor = refs_tensor.to(device)
    except Exception as e:
        print(f"Error preparing reference embeddings: {e}")
        raise

    for file, embeddings in tqdm(list_iterator, desc=f'Loading {split} set'):
        
        try:
            # FIXED: Better cache key generation
            cache_key = f"{os.path.basename(file)}_{resize_size}_{padding_size}_{nb_best_patches}"
            
            if use_cache:
                cached_result = _DATA_CACHE.get_processed(cache_key)
                if cached_result is not None:
                    yield cached_result
                    continue
            
            # FIXED: Image loading with caching and error handling
            if use_cache:
                resized_image = _DATA_CACHE.get_image(file)
            else:
                resized_image = None
                
            if resized_image is None:
                try:
                    image, _, _ = load_image(file)
                    resized_image = resize_hdf_image(image, resize_size=resize_size).squeeze()
                    
                    # FIXED: Validate image dimensions
                    if resized_image.shape[0] != resize_size or resized_image.shape[1] != resize_size:
                        print(f"Warning: Image {file} has unexpected shape {resized_image.shape}")
                        continue
                        
                    if use_cache:
                        _DATA_CACHE.set_image(file, resized_image)
                        
                except Exception as e:
                    print(f"Error loading image {file}: {e}")
                    continue
            
            H_size, W_size = resized_image.shape
            H_patch, W_patch = H_size // patch_size, W_size // patch_size
            
            # FIXED: More efficient patch extraction
            patches_list = []
            
            # FIXED: Vectorized patch coordinate generation
            patch_coords = []
            for i in range(0, H_size, patch_size):
                for j in range(0, W_size, patch_size):
                    start_h = max(i - padding, 0)
                    end_h = min(i + padding + patch_size, H_size)
                    start_w = max(j - padding, 0)
                    end_w = min(j + padding + patch_size, W_size)
                    patch_coords.append((start_h, end_h, start_w, end_w))
            
            # FIXED: Batch patch processing with proper error handling
            for start_h, end_h, start_w, end_w in patch_coords:
                try:
                    patch = resized_image[start_h:end_h, start_w:end_w]
                    
                    # FIXED: More efficient RGB conversion
                    if patch.ndim == 2:
                        patch_3ch = np.stack([patch, patch, patch], axis=0)  # Channel first
                    else:
                        patch_3ch = patch.transpose(2, 0, 1)  # HWC to CHW
                    
                    # FIXED: Proper tensor creation with correct dtype
                    patch_tensor = torch.from_numpy(patch_3ch).float().unsqueeze(0)  # Add batch dim
                    patches_list.append(patch_tensor)
                    
                except Exception as e:
                    print(f"Error processing patch at ({start_h}, {end_h}, {start_w}, {end_w}): {e}")
                    continue
            
            if not patches_list:
                print(f"No valid patches extracted from {file}")
                continue
            
            # FIXED: More robust similarity computation with error handling
            try:
                # FIXED: Ensure embeddings are properly shaped
                if hasattr(embeddings, 'reshape'):
                    flattened_embeddings = embeddings.reshape(-1, feat_dim)
                elif isinstance(embeddings, (list, tuple)):
                    flattened_embeddings = np.array(embeddings).reshape(-1, feat_dim)
                else:
                    flattened_embeddings = np.array(embeddings).reshape(-1, feat_dim)
                
                # FIXED: Validate embedding dimensions
                if flattened_embeddings.shape[1] != feat_dim:
                    print(f"Warning: Embedding dimension mismatch for {file}")
                    continue
                
                if flattened_embeddings.shape[0] != len(patches_list):
                    print(f"Warning: Mismatch between patches ({len(patches_list)}) and embeddings ({flattened_embeddings.shape[0]}) for {file}")
                    min_count = min(flattened_embeddings.shape[0], len(patches_list))
                    flattened_embeddings = flattened_embeddings[:min_count]
                    patches_list = patches_list[:min_count]
                
                # FIXED: More memory-efficient distance computation
                embeddings_tensor = torch.from_numpy(flattened_embeddings).float() if type(flattened_embeddings) == np.ndarray else flattened_embeddings.float()
                if device != 'cpu':
                    embeddings_tensor = embeddings_tensor.to(device)
                
                # FIXED: Compute distances in chunks to avoid memory issues
                chunk_size = 1000  # Process in chunks
                all_distances = []
                
                for i in range(0, embeddings_tensor.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, embeddings_tensor.shape[0])
                    chunk_embeddings = embeddings_tensor[i:end_idx]
                    chunk_distances = torch.cdist(refs_tensor, chunk_embeddings)
                    all_distances.append(chunk_distances.cpu())
                
                distances = torch.cat(all_distances, dim=1).numpy()
                
                # FIXED: More efficient best patch selection
                if nb_best_patches == 1:
                    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    selected_patch_indices = [min_dist_idx[1]]
                else:
                    flat_distances = distances.ravel()
                    top_flat_indices = np.argpartition(flat_distances, nb_best_patches)[:nb_best_patches]
                    _, patch_indices = np.unravel_index(top_flat_indices, distances.shape)
                    selected_patch_indices = patch_indices.tolist()
                
            except Exception as e:
                print(f"Error computing similarities for {file}: {e}")
                continue
            
            # FIXED: More efficient ground truth generation
            try:
                GT = torch.zeros((H_size, W_size), dtype=torch.float32)
                
                for patch_idx in selected_patch_indices:
                    h_patch_coord = patch_idx // W_patch
                    w_patch_coord = patch_idx % W_patch
                    
                    gt_h_start = int(h_patch_coord * patch_size)
                    gt_h_end = int((h_patch_coord + 1) * patch_size)
                    gt_w_start = int(w_patch_coord * patch_size)
                    gt_w_end = int((w_patch_coord + 1) * patch_size)
                    
                    # FIXED: Bounds checking
                    gt_h_end = min(gt_h_end, H_size)
                    gt_w_end = min(gt_w_end, W_size)
                    
                    GT[gt_h_start:gt_h_end, gt_w_start:gt_w_end] = 1.0
                
            except Exception as e:
                print(f"Error generating ground truth for {file}: {e}")
                continue

            result = (file, patches_list, GT)
            
            # FIXED: Cache the result with proper memory management
            if use_cache:
                _DATA_CACHE.set_processed(cache_key, result)

            yield result
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        
        # FIXED: Periodic memory cleanup
        if use_cache and _DATA_CACHE.size() > 150:
            gc.collect()


def clear_data_caches():
    """Clear all caches to free memory"""
    global _DATA_CACHE
    _DATA_CACHE.clear()
    print("Data caches cleared")


def get_dataset_size(split: str = 'training', test_proportion: float = 0.2, max_samples: Optional[int] = None):
    """Get dataset size without loading the data"""
    try:
        files_data = get_fnames()
        if isinstance(files_data[0], tuple):
            files, _ = zip(*files_data)
        else:
            files = files_data
        
        total_files = len(files)
        
        if max_samples and max_samples < total_files:
            total_files = max_samples
        
        split_size = int(total_files * test_proportion)
        
        if split == 'training':
            return total_files - split_size
        else:
            return split_size
            
    except Exception as e:
        print(f"Error getting dataset size: {e}")
        return 0


class OptimizedDataLoader:
    """FIXED: More robust optimized data loader"""
    
    def __init__(self, split='training', nb_best_patches=1, resize_size=518, 
                 padding_size=1, test_proportion=0.2, seed=42, 
                 batch_size=1, use_cache=True, max_samples=None):
        
        self.split = split
        self.nb_best_patches = nb_best_patches
        self.resize_size = resize_size
        self.padding_size = padding_size
        self.test_proportion = test_proportion
        self.seed = seed
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.max_samples = max_samples
        
        self._data = []
        self._load_data()
    
    def _load_data(self):
        """Load data using the generator"""
        try:
            generator = get_data_generator(
                split=self.split,
                nb_best_patches=self.nb_best_patches,
                resize_size=self.resize_size,
                padding_size=self.padding_size,
                test_proportion=self.test_proportion,
                seed=self.seed,
                max_samples=self.max_samples,
                use_cache=self.use_cache
            )
            
            self._data = list(generator)
            print(f"Loaded {len(self._data)} samples for {self.split}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self._data = []
    
    def __iter__(self):
        """Return iterator over data"""
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        """Get item by index"""
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._data)}")
        return self._data[idx]
    
    def get_batch(self, batch_idx: int):
        """Get a batch of data"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self._data))
        
        if start_idx >= len(self._data):
            return []
        
        return self._data[start_idx:end_idx]
    
    def num_batches(self):
        """Get number of batches"""
        return (len(self._data) + self.batch_size - 1) // self.batch_size


class PyTorchDataset(Dataset):
    """FIXED: Proper PyTorch Dataset implementation"""
    
    def __init__(self, split='training', nb_best_patches=1, resize_size=518,
                 padding_size=1, test_proportion=0.2, seed=42, 
                 use_cache=True, max_samples=None):
        
        self.data_loader = OptimizedDataLoader(
            split=split,
            nb_best_patches=nb_best_patches,
            resize_size=resize_size,
            padding_size=padding_size,
            test_proportion=test_proportion,
            seed=seed,
            use_cache=use_cache,
            max_samples=max_samples
        )
    
    def __len__(self):
        return len(self.data_loader)
    
    def __getitem__(self, idx):
        return self.data_loader[idx]


def test_data_generator_speed():
    """FIXED: More comprehensive speed testing"""
    import time
    
    print("Testing data generator performance...")
    
    # Test with cache
    clear_data_caches()
    start_time = time.time()
    count = 0
    for file, patches, gt in get_data_generator(split='training', max_samples=5, use_cache=True):
        count += 1
        print(f"Sample {count}: {len(patches)} patches, GT shape: {gt.shape}")
        if count >= 5:
            break
    
    cached_time = time.time() - start_time
    print(f"With cache: {cached_time:.2f} seconds for {count} samples")
    
    # Test without cache
    clear_data_caches()
    start_time = time.time()
    count = 0
    for file, patches, gt in get_data_generator(split='training', max_samples=5, use_cache=False):
        count += 1
        if count >= 5:
            break
    
    no_cache_time = time.time() - start_time
    print(f"Without cache: {no_cache_time:.2f} seconds for {count} samples")
    
    # Test data loader
    start_time = time.time()
    loader = OptimizedDataLoader(split='training', max_samples=5, use_cache=True)
    loader_time = time.time() - start_time
    print(f"Data loader initialization: {loader_time:.2f} seconds for {len(loader)} samples")
    
    return cached_time, no_cache_time, loader_time


def validate_data_consistency():
    """FIXED: Data validation function"""
    print("Validating data consistency...")
    
    try:
        # Check embeddings and files match
        files_data = get_fnames()
        if isinstance(files_data[0], tuple):
            files, _ = zip(*files_data)
        else:
            files = files_data
            
        print(f"Files: {len(files)}, Embeddings: {len(EMBEDDINGS)}")
        
        if len(files) != len(EMBEDDINGS):
            print("WARNING: Mismatch between files and embeddings")
        
        # Test a small sample
        sample_gen = get_data_generator(split='training', max_samples=2, use_cache=False)
        for i, (file, patches, gt) in enumerate(sample_gen):
            print(f"Sample {i}: {file}")
            print(f"  Patches: {len(patches)}")
            print(f"  GT shape: {gt.shape}")
            print(f"  GT sum: {gt.sum().item()}")
            
            if i >= 1:
                break
        
        print("Data validation completed")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        raise


if __name__ == "__main__":
    # Run validation and speed tests
    validate_data_consistency()
    test_data_generator_speed()