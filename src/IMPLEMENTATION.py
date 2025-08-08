import torch
import numpy as np
from torch import hub
from torchvision.transforms import InterpolationMode
from napari_dinosim.dinoSim_pipeline import DinoSim_pipeline
from napari_dinosim.utils import get_img_processing_f, gaussian_kernel, torch_convolve
import os
import matplotlib.pyplot as plt

compute_device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
print(f"Using device: {compute_device}")

model_size = "small"  # Options: "small", "base", "large", "giant"
model = hub.load("facebookresearch/dinov2", f"dinov2_vit{model_size}14_reg")
model.to(compute_device)
model.eval()

resize_size = 518  # Should be multiple of model patch_size
feat_dim = 384  # Depends on model size: small=384, base=768, large=1024, giant=1536
interpolation = InterpolationMode.BILINEAR if torch.backends.mps.is_available() else InterpolationMode.BICUBIC

kernel = gaussian_kernel(size=3, sigma=1)
kernel = torch.tensor(kernel, dtype=torch.float32, device=compute_device)
filter_func = lambda x: torch_convolve(x, kernel)

pipeline = DinoSim_pipeline(
    model,
    model.patch_size,
    compute_device,
    get_img_processing_f(resize_size=resize_size, interpolation=interpolation),
    feat_dim,
    dino_image_size=resize_size
)

# 6. Prepare a list of images and their reference points
# Each entry should be (image_data, reference_points)
# where reference_points is a list of (z, x, y) coordinates
# Example with two images:

def process_image(image_path, reference_coords, threshold=0.5, save_path=None):
    """
    Process a single image with its reference coordinates.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    reference_coords : list
        List of (z, x, y) coordinates for reference points
    threshold : float
        Threshold value for segmentation (default: 0.5)
    save_path : str or None
        If provided, save the mask to this path
        
    Returns:
    --------
    np.ndarray
        The segmentation mask
    """
    # Load and preprocess image
    # This is a placeholder - replace with your actual image loading code
    # For example, using PIL or OpenCV
    from PIL import Image
    import numpy as np
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to NHWC format (batch, height, width, channels)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = img_array[np.newaxis, ..., np.newaxis]
    elif len(img_array.shape) == 3:  # RGB
        if img_array.shape[2] == 3 or img_array.shape[2] == 4:  # (h,w,3) or (h,w,4)
            img_array = img_array[np.newaxis, ..., :3]  # Add batch dim, remove alpha if present
        else:  # (n,h,w)
            img_array = img_array[..., np.newaxis]  # Add channel dim
    
    # Ensure valid dtype
    if img_array.dtype == np.uint16:
        img_array = img_array.astype(np.int32)
    
    # Reset pipeline for new image
    pipeline.delete_precomputed_embeddings()
    pipeline.delete_references()
    
    # Precompute embeddings
    crop_size = 518  # Adjust as needed
    pipeline.pre_compute_embeddings(
        img_array,
        overlap=(0, 0),
        padding=(0, 0),
        crop_shape=(crop_size, crop_size, img_array.shape[-1]),
        verbose=True,
        batch_size=1
    )
    
    # Set reference points
    pipeline.set_reference_vector(list_coords=reference_coords, filter=filter_func)
    
    # Get similarity map
    distances = pipeline.get_ds_distances_sameRef(verbose=False)
    predictions = pipeline.distance_post_processing(
        distances,
        filter_func,
        upsampling_mode="bilinear"  # or None
    )
    
    # Apply threshold
    mask = predictions < threshold
    mask = np.squeeze(mask * 255).astype(np.uint8)
    
    # Save if requested
    if save_path:
        plt.imsave(save_path, mask, cmap='gray')
        print(f"Saved mask to {save_path}")
    
    return mask

# Example usage:
image_paths = [
    '/path/to/image1.jpg',
    '/path/to/image2.jpg',
    # Add more image paths as needed
]

# Define reference points for each image
# For example, if you know specific coordinates for each image:
reference_points = [
    [(0, 100, 150)],  # Reference for image1
    [(0, 200, 250)],  # Reference for image2
    # Add more reference point sets as needed
]

# Process each image
results = {}
for i, (img_path, ref_pts) in enumerate(zip(image_paths, reference_points)):
    # Create output filename
    base_name = os.path.basename(img_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = f"output_{name_without_ext}_mask.png"
    
    print(f"Processing image {i+1}/{len(image_paths)}: {base_name}")
    mask = process_image(img_path, ref_pts, threshold=0.5, save_path=output_path)
    results[base_name] = mask
    
print("Processing complete!")