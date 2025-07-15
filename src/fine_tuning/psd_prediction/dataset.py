from sklearn.model_selection import KFold
import torch
import numpy as np
import os
import random 
from torch.utils.data import Dataset
import torch.utils.data as utils
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from timeit import default_timer as timer

from src.DinoPsd import DinoPsd_pipeline
from src.DinoPsd_utils import get_img_processing_f

from src.setup import embeddings_path, feat_dim

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING DATA:
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('-Loading embeddings')

_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
_EMBEDDINGS = _EMBEDDINGS
EMBEDDINGS=[]
for e in _EMBEDDINGS:
    EMBEDDINGS.extend(e)
EMBEDDINGS = EMBEDDINGS

REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

print('...done loading embeddings')


PSD_list, REST_list = [], []

nb_best_patches = 1

for image in tqdm(EMBEDDINGS, desc='-> Comparing embeddings to reference'):
    H_patch, W_patch, _ = image.shape  # patch grid size
    flattened_image = image.reshape(-1, feat_dim)

    # Compute similarity
    similarity_matrix = euclidean_distances(REFS, flattened_image)

    # Get top K matches
    flat_similarities = similarity_matrix.ravel()
    top_flat_indices = flat_similarities.argsort()[:nb_best_patches]

    # Map to (ref_idx, patch_idx)
    ref_indices, patch_indices = np.unravel_index(top_flat_indices, similarity_matrix.shape)

    # Save top and rest patches
    PSD_list.extend(flattened_image[patch_indices])

    mask = np.ones(flattened_image.shape[0], dtype=bool)
    mask[np.unique(patch_indices)] = False
    REST_list.extend(flattened_image[mask])

assert len(PSD_list) + len(REST_list) == len(EMBEDDINGS) * H_patch * W_patch

PSD = torch.from_numpy(np.array(PSD_list))

mean_psd_embedding = np.mean(REFS, axis=0)

print('-Sorting rest embeddings...(this may take a while - check memory usage)')
start = timer()
_REST = np.array(REST_list)

distances = euclidean_distances(mean_psd_embedding[None], _REST)
idx = distances.argsort()
REST = _REST[idx, :].squeeze()
REST = torch.from_numpy(REST)

end = timer()
print(f'...done sorting rest embeddings - it took {end-start:.2f} seconds')

PSD_LABELS = torch.ones(PSD.shape[0])

REST_LABELS = torch.zeros((REST.shape[0]))


LABELLED_PSD = list(zip(PSD, PSD_LABELS))

LABELLED_REST = list(zip(REST, REST_LABELS))

print(f'-PSD shape: {PSD.shape}, Rest shape: {REST.shape}')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEFINING CUSTOM DATASETS AND DATASET GENERATORS:
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Custom_Dataset(Dataset):
    def __init__(self,
                 LABELLED_PSD: list = LABELLED_PSD,
                 LABELLED_REST: list = LABELLED_REST,
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
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
            
            yield train_loader, test_loader, training_dataset.get_dataset_info()

        
    else:
        n_splits =  len(LABELLED_REST_SET) // len(LABELLED_PSD_SET)

        for k in range(n_splits//10):
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
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
                
            yield train_loader, test_loader, training_dataset.get_dataset_info()


'''
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
