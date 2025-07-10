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


print('-Loading embeddings')

_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
_EMBEDDINGS = _EMBEDDINGS
EMBEDDINGS=[]
for e in _EMBEDDINGS:
    EMBEDDINGS.extend(e)


REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

print('...done loading embeddings')


PSD_list, REST_list = [], []

nb_best_patches = 5

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

#assert len(PSD_list) + len(REST_list) == len(EMBEDDINGS) * H_patch * W_patch

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

dataset_bias = 2

class Custom_Detection_Dataset(Dataset):
    def __init__(self, 
                 set_type,
                 test_proportion,
                 n):
        
        assert set_type in {'training', 'test'}, 'set_type must be either training or test'
        
        self.set_type = set_type
        self.test_proportion = test_proportion

        self.psd = LABELLED_PSD 
        self.len_psd = len(self.psd) // dataset_bias
        
        self.rest = LABELLED_REST[self.len_psd * n :self.len_psd * (n+1)]

        self.DATASET = self.psd + self.rest
        
        random.shuffle(self.DATASET)
        self.SPLIT = int(len(self.DATASET) * self.test_proportion)
        self.TRAINING_SET = self.DATASET[self.SPLIT:]
        self.TEST_SET = self.DATASET[:self.SPLIT]

        if set_type == 'training':
            self.data = self.TRAINING_SET
        elif set_type == 'test':
            self.data = self.TEST_SET

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_batch_size, test_batch_size = 50, 50

n_splits = (len(LABELLED_REST) // len(LABELLED_PSD)) * dataset_bias

print(f'-Number of splits: {n_splits}')

def cross_validation_datasets_generator(test_proportion):
    for k in tqdm(range(n_splits // 10), desc='Creating datasets'):
        
        training_dataset = Custom_Detection_Dataset(set_type='training', test_proportion=test_proportion, n=k) 
        test_dataset = Custom_Detection_Dataset(set_type='test', test_proportion=test_proportion, n=k)
        
        yield utils.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True), utils.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)