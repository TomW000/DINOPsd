import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import product
from tifffile import imread

from .DinoPsd_utils import get_img_processing_f

import tifffile
import gc

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import h5py
from torch.nn import functional as F
import torchvision.transforms.v2.functional as T
import torchvision.transforms.v2 as Trans
from torchvision import transforms
import torchvision
from tqdm.notebook import tqdm
import matplotlib.patches as patches
from torch import nn
import torchvision.transforms as Trans
from sklearn.decomposition import PCA
import sklearn.neighbors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import pandas as pd
import seaborn as sns
from collections import Counter
from typing import Union
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from math import ceil, floor
from itertools import chain
from random import sample

import platform

OS = platform.system()

if OS == "Linux":
    dataset_path = '/home/tomwelch/Cambridge/Datasets/neurotransmitter_data'
    embeddings_path = '/home/tomwelch/Cambridge/Embeddings'
    model_weights_path = '/home/tomwelch/Cambridge/model_weights'
    save_path ='/home/tomwelch/Cambridge/Output' 

else:
    dataset_path = '/Users/tomw/Documents/MVA/Internship/Cambridge/Datasets/neurotransmitter_data'
    embeddings_path = '/Users/tomw/Documents/MVA/Internship/Cambridge/Embeddings'
    model_weights_path = '/Users/tomw/Documents/MVA/Internship/Cambridge/model_weights'


dates = sorted(glob(os.path.join(dataset_path, '*')))
neurotransmitters = sorted(list(map(lambda x: os.path.basename(os.path.normpath(x)), glob(os.path.join(dates[0], '*'))))) #@param {type:"string"} 

upsample = "bilinear" #@param {type:"string", options:["bilinear", "Nearest Neighbor", "None"], value-map:{bilinear:"bilinear", "Nearest Neighbor": "nearest", None:None}}
crop_shape = (518,518,1) #@param {type:"raw"}


curated_idx = [21,22,25,28,30,42,43,44,51,67,
               600,601,615,617,618,621,623,625,635,636,
               1230,1244,1256,1262,1264,1273,1364,1376,1408,1432,
               1801,1803,1815,1823,1830,1853,1858,1865,1869,1877,
               2410,2417,2418,2435,2442,2444,2446,2453,2455,2458,
               3013,3015,3026,3029,3032,3040,3044,3050,3059,3061]


few_shot_transforms = [Trans.ToTensor(),       
                       Trans.Compose([Trans.ToTensor(), Trans.ColorJitter(0.4, 0.4, 0.4)]),  
                       Trans.Compose([Trans.ToTensor(), Trans.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])
                       ]

#@markdown ### Model Input Settings
#@markdown Should be multiple of model patch_size
resize_size = 518 #@param {type:"integer"} #TODO: Try other values

device = torch.device('cuda' if torch.cuda.is_available() 
                      #else 'mps' if torch.mps.is_available()
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

'''
few_shot = DinoPsd_pipeline(model, 
                            model.patch_size, 
                            device, 
                            get_img_processing_f(resize_size),
                            feat_dim, 
                            dino_image_size=resize_size )
print("Model loaded")
'''

indices = [
    [1,3,6,8,9],
    [1,2,4,5,6],
    [0,1,2,4,5],
    [1,2,3,6,7],
    [1,2,5,6,8],
    [0,6,10,11,14]
    ]

coords = [
    [(69,63.5),(68,61),(83,57),(76,62),(60,63)],
    [(66,62),(58.5,64),(64,60),(62.5,65),(64,71)],
    [(65,67),(72,60),(63,72),(60,67),(69,66.5)],
    [(65,66),(64,71),(62,58.5),(62,68),(69,55)],
    [(66,60),(60,70),(61,66.6),(58.5,63.5),(62.5,70.5)],
    [(63,73),(58,69),(60,69),(66,64),(62,71)]
    ]