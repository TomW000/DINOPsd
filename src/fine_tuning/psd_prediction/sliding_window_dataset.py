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
EMBEDDINGS = EMBEDDINGS

REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

print('...done loading embeddings')

nb_best_patches = 1

def get_data_generator(split: str = 'training', 
                       nb_best_patches: int = 1,
                       resize_size:int = 518, 
                       padding_size:int = 1, 
                       test_proportion:float = 0.2,
                       seed:int = 42): # padding_size should be an odd number
    
    patch_size = model.patch_size # type: ignore 

    assert resize_size % patch_size == 0, f'crop size must be a multiple of patch_size = {patch_size}'

    files, _ = zip(*get_fnames())

    random.seed(seed)

    DATASET = list(zip(files, EMBEDDINGS))
    random.shuffle(DATASET)
    SPLIT = int(len(DATASET) * test_proportion)
    TRAINING_SET = DATASET[SPLIT:]
    TEST_SET = DATASET[:SPLIT]

    list_iterator = TRAINING_SET if split == 'training' else TEST_SET

    for file, embeddings in tqdm(list_iterator, total=len(list_iterator), desc='Loading patches'):
        PATCHES = []
        image, _, _ = load_image(file)
        resized_image = resize_hdf_image(image, resize_size=resize_size).squeeze()
        H_size, W_size = resized_image.shape
        H_patch, W_patch = H_size // patch_size, W_size // patch_size
        
        padding = padding_size * patch_size
        
        for i in range(0, H_size, patch_size):
            start_h = max(i - padding, 0)
            end_h = min(i + padding + patch_size, H_size)
            for j in range(0, W_size, patch_size):
                start_w = max(j - padding, 0)
                end_w = min(j + padding + patch_size, W_size)
                
                patch = resized_image[start_h:end_h, start_w:end_w][...,None]
                stack = np.concatenate([patch, patch, patch], axis=2)
                stack = stack.transpose(2,0,1)[None]
                stack = torch.from_numpy(stack)
                PATCHES.append(stack)
        
        
        flattened_embeddings = embeddings.reshape(-1, feat_dim)

        # Compute similarity
        similarity_matrix = euclidean_distances(REFS, flattened_embeddings)

        # Get top K matches
        flat_similarities = similarity_matrix.ravel()
        top_flat_indices = flat_similarities.argsort()[:nb_best_patches]

        # Map to (ref_idx, patch_idx)
        ref_indices, patch_indices = np.unravel_index(top_flat_indices, similarity_matrix.shape)

        h_patch_coord = patch_indices // W_patch
        w_patch_coord = patch_indices % W_patch
        
        gt_h_start_idx = (h_patch_coord * patch_size).item()
        gt_h_end_idx = ((h_patch_coord + 1) * patch_size).item()

        gt_w_start_idx = (w_patch_coord * patch_size).item()
        gt_w_end_idx = ((w_patch_coord + 1) * patch_size).item()

        GT = torch.zeros((H_size, W_size))
        GT[gt_h_start_idx:gt_h_end_idx, gt_w_start_idx:gt_w_end_idx] = torch.ones((patch_size, patch_size))

        yield PATCHES, GT