from glob import glob
import os
import matplotlib.pyplot as plt
import torch 
import torchvision
from torchvision import transforms
import numpy as np
import tifffile

from .setup import patch_size, resize_size

def load_image(file):
    _, file_extension = os.path.splitext(file)
    if file_extension == '.tif' or file_extension == '.tiff':
        img = tifffile.imread(file) # type: ignore
    elif file_extension == '.hdf5':
        with h5py.File(file, 'r') as f: # type: ignore
            image = f['volumes/raw'][:] # type: ignore
    else:
        raise ValueError(f'Unknown file extension: {file_extension}')
    return image

def resize_image(img, resize_size=resize_size):
    image = torch.from_numpy(img).unsqueeze(0) if type(img) == np.ndarray else img.unsqueeze(0)
    resized_image = transforms.Resize(size=(resize_size, resize_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(image)
    resized_image = resized_image.permute(1,2,0).numpy()
    return resized_image

def display_image(image, resize_size=resize_size):
    resized_image = resize_image(image, resize_size)
    plt.figure(figsize=(10,10), dpi=100)
    plt.imshow(resized_image, cmap='grey')
    plt.xticks([i for i in range(0,resize_size, patch_size)])
    plt.yticks([i for i in range(0,resize_size, patch_size)])
    plt.grid(True, color='r', linewidth=1)
    plt.xticks(rotation = -90)
    def format_coord(x, y):
        return f"Image coords: ({x:.0f}, {y:.0f})"
    
    plt.gca().format_coord = format_coord
    plt.show()
