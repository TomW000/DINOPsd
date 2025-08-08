from .setup import torch, Trans, torchvision, np, h5py, plt, product, sns
from .setup import resize_size, feat_dim, model, device
from .DinoPsd import DinoPsd_pipeline
from .DinoPsd_utils import get_img_processing_f


few_shot = DinoPsd_pipeline(model, 
                            model.patch_size, 
                            device, 
                            get_img_processing_f(resize_size),
                            feat_dim, 
                            dino_image_size=resize_size )


def get_threshold(array, proportion):
    threshold = torch.quantile(torch.from_numpy(array), torch.tensor([proportion]), interpolation='linear')
    return threshold.item()


def N_pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    return y_out


def add_context_to_embeddings(path, delta): # delta should be an even number 
    with h5py.File(path) as f:

        pre, _ = f['annotations/locations'][:]/8
        _, _, z = pre.astype(int)
        slices_list = [torch.from_numpy(f['volumes/raw'][:][np.newaxis,:,:,z+i]) for i in np.arange(-delta,delta+1)] # (nb_slices, 130, 130, 1)

        resized_slices_list = np.array([Trans.Resize(size=resize_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(img).numpy() for img in slices_list]).transpose(0,2,3,1)

        nb_slices = len(resized_slices_list)

        few_shot.pre_compute_embeddings(
        resized_slices_list,  # Pass numpy array of images
        overlap= (0, 0), #(0.5, 0.5),
        padding=(0, 0),
        crop_shape=(518, 518, 1),
        verbose=True,
        batch_size=nb_slices
        )
        
        few_shot.set_reference_vector([(0,273,245),])
        augmented_ref = few_shot.get_reference_embedding()

        latents = np.array(few_shot.get_embeddings()) # should be of shape (nb_slices, nb_patches, embedding_dim)

        nb_patches = int((resize_size/14)**2)
        reshaped_latents = latents.reshape(nb_slices, nb_patches, feat_dim)
        
        few_shot.delete_precomputed_embeddings()
        few_shot.delete_references()
        
        weighted_embeddings_list = []
        for j in range(nb_patches):
            strip = reshaped_latents[:,j,:]
            vectors = [strip[i,:] for i in range(nb_slices)]
            
            positions = []
            positions = [i for i in range(nb_slices//2+1)][::-1]
            positions.extend(positions[:nb_slices//2][::-1])
            
            weighted_positions = np.stack([N_pdf(np.array(positions)) for _ in range(feat_dim)]).T
            weighted_vectors = np.multiply(vectors, weighted_positions) # should be (nb_slices, embedding_dim)) # TODO: ADD ASSERT
            weighted_embedding = np.mean(weighted_vectors, axis = 0) # should be (1, embedding_dim)
            weighted_embeddings_list.append(weighted_embedding)
            
        weighted_embeddings = np.vstack(weighted_embeddings_list) # (nb_patches, embedding_dim)
        
        return weighted_embeddings, augmented_ref

def resize_hdf_image(image, resize_size):
    image = torch.from_numpy(image)
    resized_image = Trans.Resize(size=(resize_size, resize_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(image).permute(1,2,0).numpy()
    return resized_image

def resize_tiff_image(img, resize_size):
    image = torch.from_numpy(img).unsqueeze(0) if type(img) == np.ndarray else img.unsqueeze(0)
    resized_image = Trans.Resize(size=(resize_size, resize_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(image).permute(1,2,0).numpy()
    return resized_image

def display_hdf_image_grid(image, resize_size):
    resized_image = resize_hdf_image(image, resize_size)
    plt.figure(figsize=(10,10), dpi=100)
    plt.imshow(resized_image, cmap='grey')
    plt.xticks([i for i in range(0,resize_size, model.patch_size)])
    plt.yticks([i for i in range(0,resize_size, model.patch_size)])
    plt.grid(True, color='r', linewidth=1)
    plt.xticks(rotation = -90)
    #plt.scatter((resize_size/130)*y, (resize_size/130)*x, marker='x', c='red', label=f'Reference:x,y={(resize_size/130)*y, (resize_size/130)*x}')
    #plt.legend(loc='upper right')
    def format_coord(x, y):
        return f"Image coords: ({x:.0f}, {y:.0f})"
    
    plt.gca().format_coord = format_coord
    plt.show()

def display_tiff_image_grid(path, index):
    from tifffile import imread
    img = imread(path)[index]
    resized_image = resize_tiff_image(img)
    plt.figure(figsize=(10,10), dpi=100)
    plt.imshow(resized_image, cmap='grey')
    plt.xticks([i for i in range(0,resize_size, model.patch_size)])
    plt.yticks([i for i in range(0,resize_size, model.patch_size)])
    plt.grid(True, color='r', linewidth=1)
    plt.xticks(rotation = -90)
    plt.show()

def compute_similarity_matrix(reference, embeddings):
    
    nb_patches_per_dim = int((resize_size/14))
    similarity_list = []
    embeddings = embeddings.reshape(-1, feat_dim)
    for patch in embeddings:
        similarity_list.append(np.dot(reference, patch)/(np.linalg.norm(reference)*np.linalg.norm(patch)))
    patch_similarities = np.array(similarity_list).reshape(nb_patches_per_dim, nb_patches_per_dim)
    return patch_similarities

def get_augmented_coordinates(reference_coordinates: list[int]):
    
    possibilities = list(product((14,-14, 0),repeat=2))
    augmentations = [np.add(reference_coordinates, possibility) for possibility in possibilities]
    augmented_ref_coordinates = [(0, a[0], a[1]) for a in augmentations]
    return augmented_ref_coordinates

def get_resized_volume(path):

    from tifffile import imread
    volume = imread(path)
    nb_slices = volume.shape[0]
    volume_list = []
    for i in range(nb_slices):
        resized_slice = resize_tiff_image(volume[i])
        volume_list.append(resized_slice)
    gt = np.array(volume_list)
    return gt

def get_resized_ground_truth(path):

    from tifffile import imread
    volume = imread(path)
    nb_slices = volume.shape[0]
    gt_list = []
    for i in range(nb_slices):
        slice = torch.from_numpy(volume[i]).unsqueeze(0)
        resized_image = Trans.Resize(size=(37, 37), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(slice).permute(1,2,0).numpy()
        gt_list.append(resized_image)
    gt = np.array(gt_list).astype(np.uint8)
    return gt