import torch
import numpy as np
import os
import random 
from torch.utils.data import Dataset
import torch.utils.data as utils

from src.DinoPsd import DinoPsd_pipeline
from src.DinoPsd_utils import get_img_processing_f
from src.fine_tuning.compute_embeddings import compute_embeddings

from src.setup import resize_size, embeddings_path, neurotransmitters


device = torch.device('cuda' if torch.cuda.is_available() 
                      else 'mps' if torch.mps.is_available()
                      else 'cpu')
print("Device:", device)

# select model size
model_size = 'small' #@param {type:"string", options:["small", "base", "large", "giant"]}

model_dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
assert model_size in model_dims, f'Invalid model size: ({model_size})'
model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size[0]}14_reg')
model.to(device)
model.eval()

feat_dim = model_dims[model_size]

few_shot = DinoPsd_pipeline(model, 
                            model.patch_size, 
                            device, 
                            get_img_processing_f(resize_size),
                            feat_dim, 
                            dino_image_size=resize_size )


DATA = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
print('Done loading embeddings')

#TODO:
filtering = False

if filtering:

    LABELS = np.hstack([[neuro]*600 for neuro in neurotransmitters]).reshape(-1,1)

    REFS = torch.load(os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt'), weights_only=False)

    DATASET = few_shot.get_d_closest_elements(embeddings = DATA, 
                                            reference_emb = torch.from_numpy(REFS))

else:
    
    LABELS = np.hstack([[neuro]*int((resize_size/14)**2 * 600) for neuro in neurotransmitters]).reshape(-1, 1)
    
    DATA = torch.cat(DATA)
    DATA = DATA.reshape(-1, feat_dim)
    
    DATASET = list(zip(DATA, LABELS))

random.shuffle(DATASET)


test_proportion = 0.2


SPLIT = int(len(DATASET)*test_proportion)
TRAINING_SET = DATASET[SPLIT:]
TEST_SET = DATASET[:SPLIT]

one_hot_neurotransmitters = np.eye(len(neurotransmitters))


class Custom_LP_Dataset(Dataset):
    def __init__(self, 
                 set):
        if set == 'training':
            self.data = TRAINING_SET
        else:
            self.data = TEST_SET

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding, label = self.data[idx]
        label_idx = neurotransmitters.index(label[0])
        return embedding, one_hot_neurotransmitters[label_idx]
    
    
train_batch_size, test_batch_size = 50, 50

training_dataset = Custom_LP_Dataset('training') 
test_dataset = Custom_LP_Dataset('test')

training_loader = utils.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)