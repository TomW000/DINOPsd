import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .training import head_weights
from src.setup import embeddings_path, feat_dim
from src.perso_utils import get_fnames, load_image
from src.analysis_utils import resize_hdf_image


classification_head = torch.load(head_weights)

IMAGE_EMBEDDINGS = torch.load(os.path.join(embeddings_path, 'small_dataset_embs_518.pt'))
image_dim = IMAGE_EMBEDDINGS[0].shape[1]

file_names, _ = zip(*get_fnames())

DATA = zip(file_names, IMAGE_EMBEDDINGS) 

for file_name, image in DATA:
    flattened_image = image.reshape(1, feat_dim)
    image_prediction = []
    for patch in flattened_image:
        prediction = classification_head(patch)
        prediction = torch.argmax(prediction)
        image_prediction.append(prediction)

    mask = np.array(image_prediction).reshape(image_dim, image_dim)
    
    h, w = mask.shape
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    initial_image = load_image(file_name)[0]
    resized_image = resize_hdf_image(initial_image)

    ax.imshow(resized_image, cmap='gray', extent=[0, h, w, 0])

    # Overlay the heatmap
    sns.heatmap(
        mask,
        cmap='coolwarm',
        alpha=0.5,             # Make the heatmap semi-transparent
        ax=ax,
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )

    plt.title("Patch Similarities with augmented reference coordinates")
    plt.tight_layout()
    plt.show()