from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as utils
from random import sample
import torch
import torchvision

from src.setup import neurotransmitters
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image

def get_dataset(resize_size:int, window_size:int = 3):
    
    patch_size = model.patch_size
    
    assert resize_size % patch_size == 0, f'crop size must be a multiple of patch_size = {patch_size}'

    files, _ = zip(*get_fnames())

    IMAGES = []
    for file in tqdm(files, desc='Loading images'):
        image, _, _ = load_image(file)
        resized_image = resize_hdf_image(image, resize_size=resize_size).squeeze()
        nb_x, nb_y = resized_image.shape // patch_size
        
        for i in range(0, nb_x, patch_size):
            start_x = i
            end_x = i + window_size
            for j in range(0, nb_y, patch_size):
                start_y = j
                end_y = j + window_size
                
                patch = resized_image[start_x:end_x, start_y:end_y]
        
        
        
        stack = np.concatenate([cropped_im, cropped_im, cropped_im], axis=2)
        IMAGES.append(stack)
    IMAGES = np.array(IMAGES).transpose(0,3,1,2)

    FT_LABELS = np.hstack([[neuro]*600 for neuro in neurotransmitters]).reshape(-1,1)

    FT_DATASET = list(zip(IMAGES, FT_LABELS))

    FT_DATASET = sample(FT_DATASET, len(FT_DATASET))

    test_proportion = 0.2

    FT_SPLIT = int(len(FT_DATASET)*test_proportion)
    FT_TRAINING_SET = FT_DATASET[FT_SPLIT:]
    FT_TEST_SET = FT_DATASET[:FT_SPLIT]
    class Custom_FT_Dataset(Dataset):
        def __init__(self, 
                    set):
            if set == 'training':
                self.data = FT_TRAINING_SET
            else:
                self.data = FT_TEST_SET

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image, label = self.data[idx]
            label_idx = neurotransmitters.index(label[0])
            return image, one_hot_neurotransmitters[label_idx]
        
    ft_train_batch_size, ft_test_batch_size = 1, 1

    ft_training_dataset = Custom_FT_Dataset('training') 
    ft_test_dataset = Custom_FT_Dataset('test')

    ft_training_loader = utils.DataLoader(ft_training_dataset, batch_size=ft_train_batch_size, shuffle=True, pin_memory=True)
    ft_test_loader = utils.DataLoader(ft_test_dataset, batch_size=ft_test_batch_size, shuffle=True, pin_memory=True)
    
    return ft_training_loader, ft_test_loader