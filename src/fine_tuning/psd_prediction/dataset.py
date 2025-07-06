from sklearn.model_selection import KFold
import torch
import numpy as np
import os
import random 
from torch.utils.data import Dataset
import torch.utils.data as utils
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from src.DinoPsd import DinoPsd_pipeline
from src.DinoPsd_utils import get_img_processing_f

from src.setup import embeddings_path, feat_dim

_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
_EMBEDDINGS = np.vstack(_EMBEDDINGS)[:600]

nb_columns = _EMBEDDINGS.shape[0]
columns = np.random.choice(nb_columns, size=nb_columns, replace=False)

EMBEDDINGS = _EMBEDDINGS[columns, :]

assert EMBEDDINGS.shape == _EMBEDDINGS.shape

print('Done loading embeddings')


REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

print('Done loading reference embeddings')


PSD_list, REST_list = [], []

nb_best_patches = 10

for image in tqdm(EMBEDDINGS, desc='Comparing embeddings to reference'):
    flattened_image = image.reshape(-1, feat_dim)
    similarity_matrix = cosine_similarity(REFS, flattened_image)
    flat_similarities = np.unique(similarity_matrix.ravel())
    top_10_flat_indices = flat_similarities.argsort()[-nb_best_patches:][::-1]
    best_indices = np.unravel_index(top_10_flat_indices, similarity_matrix.shape)[1]
    PSD_list.extend(flattened_image[best_indices])
    REST_list.extend(np.delete(flattened_image, best_indices, axis=0))


PSD = np.array(PSD_list)
REST = np.array(REST_list)


PSD_LABELS = np.ones((PSD.shape[0]))

REST_LABELS = np.zeros((REST.shape[0]))


LABELLED_PSD = list(zip(PSD, PSD_LABELS))

LABELLED_REST = list(zip(REST, REST_LABELS))


class Custom_Detection_Dataset(Dataset):
    def __init__(self, 
                 set_type,
                 test_proportion,
                 n):
        
        assert set_type in {'training', 'test'}, 'set_type must be either training or test'
        
        self.set_type = set_type
        self.test_proportion = test_proportion

        self.psd = LABELLED_PSD 
        self.len_psd = len(self.psd) // 2
        
        self.rest = LABELLED_REST[self.len_psd * n :self.len_psd * (n+1)]

        self.DATASET = self.psd + self.rest
        
        self.SPLIT = int(len(self.DATASET) * self.test_proportion)
        self.TRAINING_SET = self.DATASET[self.SPLIT:]
        self.TEST_SET = self.DATASET[:self.SPLIT]
        
        print(f'Dataset length: {len(self.TRAINING_SET)}, {len(self.TEST_SET)}')
        
        if set_type == 'training':
            self.data = self.TRAINING_SET
        elif set_type == 'test':
            self.data = self.TEST_SET

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_batch_size, test_batch_size = 1, 1

n_splits = (len(LABELLED_REST) // len(LABELLED_PSD)) * 2

print(f'Number of splits: {n_splits}')

def cross_validation_datasets_generator(test_proportion):
    for k in tqdm(range(n_splits), desc='Creating datasets'):
        
        training_dataset = Custom_Detection_Dataset(set_type='training', test_proportion=test_proportion, n=k) 
        test_dataset = Custom_Detection_Dataset(set_type='test', test_proportion=test_proportion, n=k)
        
        yield utils.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True), utils.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
