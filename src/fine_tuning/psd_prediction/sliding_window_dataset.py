from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as utils
import torch
import torchvision

#from src.setup import neurotransmitters, model
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image


# TODO: ADD EMBEDDING COMPUTATION FUNCTION AS DEFAULT FIRST STEP

import os
import random 
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from src.setup import embeddings_path, model, feat_dim

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
EMBEDDINGS = EMBEDDINGS[:2]

REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

print('...done loading embeddings')

nb_best_patches = 1

def get_dataset(nb_best_patches: int = 1,
                resize_size:int = 518, 
                padding_size:int = 1, 
                test_proportion:float = 0.2,
                batch_size:int = 1): # padding_size should be an odd number
    
    patch_size = model.patch_size # type: ignore 

    assert resize_size % patch_size == 0, f'crop size must be a multiple of patch_size = {patch_size}'

    files, _ = zip(*get_fnames())

    DATASET = []

    for k, file in tqdm(enumerate(files), desc='Loading patches'):
        PATCHES = []
        image, _, _ = load_image(file)
        resized_image = resize_hdf_image(image, resize_size=resize_size).squeeze()
        x, y = resized_image.shape
        
        H_patch, W_patch = resized_image.shape
        flattened_image = resized_image.reshape(-1, feat_dim)

        # Compute similarity
        similarity_matrix = euclidean_distances(REFS, flattened_image)

        # Get top K matches
        flat_similarities = similarity_matrix.ravel()
        top_flat_indices = flat_similarities.argsort()[:nb_best_patches]

        # Map to (ref_idx, patch_idx)
        ref_indices, patch_indices = np.unravel_index(top_flat_indices, similarity_matrix.shape)

        y_coord = patch_indices // H_patch
        x_coord = patch_indices % W_patch
        
        gt_y_start_idx = y_coord * patch_size
        gt_y_end_idx = (y_coord + 1) * patch_size

        gt_x_start_idx = x_coord * patch_size
        gt_x_end_idx = (x_coord + 1) * patch_size
        
        GT = torch.zeros((y, x))
        GT[gt_y_start_idx:gt_y_end_idx, gt_x_start_idx:gt_x_end_idx] = torch.ones((patch_size, patch_size))


        padding = padding_size * patch_size
        
        for i in range(0, x, patch_size):
            start_x = max(i - padding, 0)
            end_x = min(i + padding + patch_size, x)
            for j in range(0, y, patch_size):
                start_y = max(j - padding, 0)
                end_y = min(j + padding + patch_size, y)
                
                patch = resized_image[start_x:end_x, start_y:end_y][...,None]
                stack = np.concatenate([patch, patch, patch], axis=2)
                stack = stack.transpose(2,0,1)[None]
                PATCHES.append(stack)
                
        DATASET.append([PATCHES, GT])

    random.shuffle(DATASET)

    nb_images = len(DATASET)

    SPLIT = int(nb_images*test_proportion)
    TRAINING_SET = DATASET[SPLIT:]
    TEST_SET = DATASET[:SPLIT]

    training_dataset = Custom_Dataset(TRAINING_SET) 
    test_dataset = Custom_Dataset(TEST_SET)

    training_loader = utils.DataLoader(training_dataset, batch_size=batch_size)
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size)

    return training_loader, test_loader


class Custom_Dataset(Dataset):
    def __init__(self,
                 SET):
        self.data = SET
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]