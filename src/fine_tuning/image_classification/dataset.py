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

def get_dataset(crop_size:int):
    one_hot_neurotransmitters = np.eye(len(neurotransmitters))

    files, _ = zip(*get_fnames())

    IMAGES = []
    for file in tqdm(files, desc='Loading images'):
        im = resize_hdf_image(load_image(file)[0]).squeeze()
        cropped_im = torchvision.transforms.CenterCrop(crop_size)(torch.from_numpy(im)).unsqueeze(2)
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