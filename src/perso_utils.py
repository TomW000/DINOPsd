from .DinoPsd_utils import gaussian_kernel, torch_convolve
from .setup import *
from .setup import np, os, plt, pd, sns, Counter, tqdm, torch, h5py, glob

from .analysis_utils import *

kernel = gaussian_kernel(size=3, sigma=1)
kernel = torch.tensor(kernel, dtype=torch.float32, device=device)
filter_f = lambda x: torch_convolve(x, kernel)

resize_factor = resize_size/130
resize = lambda x: resize_factor*x

#----------------------------------------------------------------------------------------------------------------------------------------------

def get_fnames():
    files_list, labels_list = [], []
    for date in dates:
        for neuro in neurotransmitters:
            fnames = glob(os.path.join(date, neuro, '*.hdf*'))
            files_list.append(fnames)
            labels_list.append([neuro for _ in range(len(fnames))])
    files = sorted(np.concatenate(files_list), key=os.path.basename)
    labels = sorted(np.concatenate(labels_list), key=os.path.basename)
    return list(zip(files, labels))

#----------------------------------------------------------------------------------------------------------------------------------------------

def load_image(path):
    with h5py.File(path) as f:
        pre, _ = f['annotations/locations'][:]/8
        x, y, z = pre.astype(int)
        slice_volume = f['volumes/raw'][:][np.newaxis,:,:,z]
        return slice_volume, x, y

#----------------------------------------------------------------------------------------------------------------------------------------------

def get_processed_image(path):
    main_axis, norm = get_direction(path)
    norm = round(norm)
    with h5py.File(path) as f:
        pre, _ = f['annotations/locations'][:]/8
        x, y, z = [pre[i].astype(int) for i in range(3)]
        
        slice_volume = f['volumes/raw'][:]
        
        dim_x, dim_y, dim_z = slice_volume.shape
        
        if main_axis == 'x':
            y1, y2 = max(0, y-norm), min(dim_y, y+norm+1)
            z1, z2 = max(0, z-norm), min(dim_z, z+norm+1)
            return slice_volume[np.newaxis,x,:,:,np.newaxis], y, z, slice_volume[np.newaxis,x, y1:y2,z1:z2,np.newaxis], y-y1, z-z1, y2, z2
        elif main_axis == 'y':
            x1, x2 = max(0, x-norm), min(dim_x, x+norm+1)
            z1, z2 = max(0, z-norm), min(dim_z, z+norm+1)
            return slice_volume[np.newaxis,:,y,:,np.newaxis], x, z, slice_volume[np.newaxis,x1:x2, y,z1:z2,np.newaxis], x-x1, z-z1, x1, z1
        else:
            x1, x2 = max(0, x-norm), min(dim_x, x+norm+1)
            y1, y2 = max(0, y-norm), min(dim_y, y+norm+1)
            return slice_volume[np.newaxis,:,:,z,np.newaxis], x, y, slice_volume[np.newaxis,x1:x2,y1:y2,z,np.newaxis], x-x1, y-y1, x1, y1

#----------------------------------------------------------------------------------------------------------------------------------------------

def get_prediction(image: np.array,
                   coordinates: tuple):
    few_shot.delete_precomputed_embeddings()
    few_shot.delete_references()
    if not few_shot.emb_precomputed:
        few_shot.pre_compute_embeddings(image, 
                                        overlap=(0, 0),
                                        padding=(0, 0),
                                        crop_shape=crop_shape, 
                                        verbose=False,
                                        batch_size=1)

    few_shot.set_reference_vector(list_coords=[coordinates])
    distances = few_shot.get_ds_distances_sameRef(verbose=False)
    pred = few_shot.distance_post_processing(distances, filter_f, upsampling_mode=upsample)
    return pred

#----------------------------------------------------------------------------------------------------------------------------------------------

def get_bbox(predictions, threshold):
    GT = (predictions[0,...] < threshold).astype(np.float32)
    bbox_1x, bbox_1y = min(np.where(GT == 1)[1])-5, min(np.where(GT == 1)[0])-5
    bbox_2x, bbox_2y = max(np.where(GT == 1)[1])+5, max(np.where(GT == 1)[0])+5
    return bbox_1x, bbox_2x, bbox_1y, bbox_2y
    #print(f"{len(failed)/predictions.shape[0]*100}% of images did not pass the threshold")
    

#----------------------------------------------------------------------------------------------------------------------------------------------

def f(predictions, threshold):
    return get_bbox(predictions, threshold)[1]

#----------------------------------------------------------------------------------------------------------------------------------------------

def get_direction(path):
    with h5py.File(path) as f:
        pre, post = f['annotations/locations'][:]/8
        x_pre, y_pre, z_pre = pre[0].astype(int), pre[1].astype(int), pre[2].astype(int)
        x_post, y_post, z_post = post[0].astype(int), post[1].astype(int), post[2].astype(int)
        direction_vector = np.array([x_pre - x_post, y_pre - y_post, z_pre - z_post])
        main_axis = np.argmax(np.abs(direction_vector))
        norm = np.linalg.norm(direction_vector)
        coord_list = ['x','y','z']
        return coord_list[main_axis], norm

#----------------------------------------------------------------------------------------------------------------------------------------------

def statistics():
    path_list, labels = get_fnames()
    neuro_stats = []
    
    for path in path_list:
        _, norm = get_direction(path)
        neuro_stats.append(norm)
    
    df = pd.DataFrame({'Neurotransmitter': labels, 'Distance': neuro_stats})
    
    plt.figure(figsize=(12,7), dpi=300)
    sns.boxplot(x='Neurotransmitter', y='Distance', data=df)
    plt.title("Pre-Post Distances per Neurotransmitter")
    plt.ylabel("Pre-Post Distance (pixel)")
    plt.xlabel("Neurotransmitter")
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------

def plot_main_axis():
    fnames = get_fnames()[0]
    dir = []
    for file in fnames:
        dir.append(get_direction(file)[0])
    
    counts = Counter(dir)
    
    plt.figure(figsize=(12,7), dpi=300)
    plt.bar(counts.keys(), counts.values())
    plt.title('Main Axis Distribution')
    plt.ylabel('Nb')
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------

def display_cropped_image(n):
    for date in dates:
            for neuro in neurotransmitters:
                fnames = glob(os.path.join(date, neuro, '*.hdf*'))
                fnames.sort()
                for file in fnames[:n]:
                    img_c, img_o, _, _ = get_processed_image(file)
                    plt.figure(figsize=(12,7), dpi=300)
                    plt.suptitle(f'{neuro.capitalize()}')
                    plt.subplot(121)
                    plt.imshow(img_o, cmap='gray')
                    plt.title(f'Original image')
                    plt.subplot(122)
                    plt.imshow(img_c, cmap='gray')
                    plt.title('Cropped image')
                    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------

def image_selection():
    i=0
    for file, label in get_fnames():
        img, x, y = load_image(file)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.scatter(x,y, label=str(i))
        plt.title(label)
        plt.legend()
        plt.show()
        i+=1

#----------------------------------------------------------------------------------------------------------------------------------------------

def get_latents(dataset_paths: list[str], 
                labels: list[str],
                k: int
                ) -> tuple[np.array]:
    
    
    latent_list = []
    
    #for file, label in tqdm(zip(dataset_paths, labels), desc=f'Computing latent vectors'):

    # Load images and prepare data
    images = np.array([load_image(file)[0] for file in dataset_paths]).transpose(0,2,3,1)
    coordinates = [(0, load_image(file)[1], load_image(file)[2]) for file in dataset_paths]

    # Pre-compute embeddings
    few_shot.pre_compute_embeddings(
        images,
        overlap=(0.5, 0.5),
        padding=(0, 0),
        crop_shape=(512, 512, 1),
        verbose=True,
        batch_size=50
    )
    
    # Set reference vectors
    few_shot.set_reference_vector(coordinates, filter=None)
    
    # Get closest elements - using the correct method name
    
    k_closest_embeddings = few_shot.get_k_closest_elements(
        k=k
    )
    k_labels = [labels for _ in range(k)]
    
    k_closest_embeddings_np = k_closest_embeddings.cpu().numpy() if isinstance(k_closest_embeddings, torch.Tensor) else k_closest_embeddings
    
    latent_list.append(k_closest_embeddings_np)
    
    # Clean up to free memory
    few_shot.delete_precomputed_embeddings()
    few_shot.delete_references()

    # Stack all embeddings and labels
    return np.vstack(latent_list), np.hstack(k_labels)  

#----------------------------------------------------------------------------------------------------------------------------------------------

def resize_coords(x):
    resize_factor = resize_size/130
    resize = lambda x: resize_factor*x
    return x

#----------------------------------------------------------------------------------------------------------------------------------------------

def compute_ref_embeddings(saved_ref_embeddings=False, 
                           embs_path=None, 
                           k=10,
                           data_aug=False):

    if saved_ref_embeddings:
        
        mean_ref = torch.load(embs_path, weights_only=False)
        return mean_ref

    else:
        files, labels = zip(*get_fnames()) 
        if data_aug:    
            nb_transformations = len(few_shot_transforms)
            
            # Preload images and metadata once
            good_images = []
            transformed_coordinates = []

            for idx in curated_idx:
                img, coord_x, coord_y = load_image(files[idx])
                good_images.append(img.transpose(1,2,0))
                transformed_coordinates.append([(0, coord_x, coord_y)] * nb_transformations)

            transformed_images = []
            for image in good_images:
                transformed = [t(image).permute(1,2,0) for t in few_shot_transforms]
                transformed_images.extend(transformed)

            for j, img in enumerate(transformed_images):
                if img.shape != torch.Size([130, 130, 1]):
                    h, w = img.shape[:2]
                    h_diff = (130 - h) // 2
                    w_diff = (130 - w) // 2
                    padded_img = torch.zeros(130, 130, 1)
                    padded_img[h_diff:h+h_diff, w_diff:w+w_diff, :] = img
                    transformed_images[j] = padded_img
                    
            batch_size = int(len(curated_idx)/len(neurotransmitters)*nb_transformations) # nb of images in per class
            good_datasets = [transformed_images[i:i+batch_size] for i in range(0,len(transformed_images),batch_size)]
            good_datasets = np.array(good_datasets)
            
            transformed_coordinates = np.vstack(transformed_coordinates)
            good_coordinates = [transformed_coordinates[i:i+batch_size] for i in range(0,len(transformed_coordinates),batch_size)]

        else:
            ref_embs_list = []
            for i, index in tqdm(enumerate(indices)):
                dataset_slice = files[i*600:(i+1)*600]
                imgs = [resize_hdf_image(load_image(dataset_slice[k])[0]) for k in index]
                coordinates = [list(map(resize, c)) for c in coords[i]]
                dataset = list(zip(imgs, coordinates))
                class_wise_embs_list = []
                for image, reference in dataset:
                    few_shot.pre_compute_embeddings(
                        image[None,:,:,:],
                        verbose=False,
                        batch_size=1
                    )
                    few_shot.set_reference_vector(get_augmented_coordinates(reference), filter=None)
                    closest_embds = few_shot.get_k_closest_elements(k=k, return_indices=False) # list of vectors
                    class_wise_embs_list.append(torch.mean(closest_embds.cpu(), dim=0)) # list of lists of vectors
                ref_embs_list.append(class_wise_embs_list) # list of lists of lists of vectors

            ref_embs = np.array([np.mean(class_closest_embs, axis=0) for class_closest_embs in ref_embs_list])
            
            #torch.save(ref_embs, os.path.join(embeddings_path, f'{model_size}_mean_ref_{resize_size}_Aug={data_aug}_k={k}'))
            
            return ref_embs