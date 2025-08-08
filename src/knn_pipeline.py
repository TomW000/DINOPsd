from .setup import neurotransmitters, model_size, device, feat_dim, resize_size, curated_idx, few_shot_transforms, embeddings_path, model
from .setup import tqdm, torch, np, os, plt, tqdm, gc, sample
from .analysis_utils import display_hdf_image_grid, resize_hdf_image, get_augmented_coordinates
from .setup import cosine_similarity, euclidean_distances, feat_dim
from .perso_utils import get_fnames, load_image, get_latents
from .DinoPsd import DinoPsd_pipeline
from .DinoPsd_utils import get_img_processing_f

device = 'cpu'

# Initialize the few-shot pipeline
few_shot = DinoPsd_pipeline(
    model,
    model.patch_size,
    device,
    get_img_processing_f(resize_size),
    feat_dim, 
    dino_image_size=resize_size
)

files, labels = zip(*get_fnames()) 

def compute_ref_embeddings(saved_ref_embeddings=False, 
                           embs_path=None, 
                           k=10,
                           data_aug=True):
    """
    Compute reference embeddings for neurotransmitter classification.
    
    Args:
        saved_ref_embeddings: Whether to load pre-computed embeddings
        embs_path: Path to saved embeddings
        k: Number of closest elements to retrieve
        data_aug: Whether to apply data augmentation
    
    Returns:
        torch.Tensor: Mean reference embeddings
    """
    
    if saved_ref_embeddings and embs_path is not None:
        try:
            mean_ref = torch.load(embs_path, weights_only=False, map_location=device)
            return mean_ref
        except FileNotFoundError:
            print(f"Warning: Could not load embeddings from {embs_path}. Computing new embeddings.")
    
    # Compute new embeddings
    if data_aug:    
        nb_transformations = len(few_shot_transforms)
        
        # Preload images and metadata once
        good_images = []
        transformed_coordinates = []

        for idx in curated_idx:
            img, coord_x, coord_y = load_image(files[idx])
            good_images.append(img.transpose(1, 2, 0))
            transformed_coordinates.append([(0, coord_x, coord_y)] * nb_transformations)

        # Apply transformations
        transformed_images = []
        for image in good_images:
            transformed = [t(image).permute(1, 2, 0) for t in few_shot_transforms]
            transformed_images.extend(transformed)

        # Ensure consistent image size (130x130x1)
        for j, img in enumerate(transformed_images):
            if img.shape != torch.Size([130, 130, 1]):
                h, w = img.shape[:2]
                h_diff = (130 - h) // 2
                w_diff = (130 - w) // 2
                padded_img = torch.zeros(130, 130, 1)
                padded_img[h_diff:h+h_diff, w_diff:w+w_diff, :] = img
                transformed_images[j] = padded_img
                
        batch_size = int(len(curated_idx) / len(neurotransmitters) * nb_transformations)
        good_datasets = [transformed_images[i:i+batch_size] for i in range(0, len(transformed_images), batch_size)]
        good_datasets = np.array(good_datasets)
        
        transformed_coordinates = np.vstack(transformed_coordinates)
        good_coordinates = [transformed_coordinates[i:i+batch_size] for i in range(0, len(transformed_coordinates), batch_size)]

    else:
        # No data augmentation
        imgs_coords = [load_image(files[idx]) for idx in curated_idx]
        imgs, xs, ys = zip(*imgs_coords)

        batch_size = int(len(curated_idx) / len(neurotransmitters))
        imgs = [imgs[i:i+batch_size] for i in range(0, len(imgs), batch_size)]
        good_datasets = np.array(imgs).transpose(0, 1, 3, 4, 2)
        
        good_coordinates = [(0, x, y) for x, y in zip(xs, ys)]
        good_coordinates = [good_coordinates[i:i+batch_size] for i in range(0, len(good_coordinates), batch_size)]
        good_coordinates = np.array(good_coordinates)

    # Process each neurotransmitter class
    filtered_latent_list = []
    
    for dataset, batch_label, coordinates in tqdm(zip(good_datasets, neurotransmitters, good_coordinates), 
                                                 desc='Processing neurotransmitters'):
        
        # Pre-compute embeddings
        few_shot.pre_compute_embeddings(
            dataset,
            overlap=(0.5, 0.5),
            padding=(0, 0),
            crop_shape=(518, 518, 1),
            verbose=True,
            batch_size=10
        )
        
        # Set reference vectors
        few_shot.set_reference_vector(coordinates, filter=None)
        
        # Get k closest elements
        close_embedding = few_shot.get_k_closest_elements(k=k)
        
        # Convert to numpy for storing
        if isinstance(close_embedding, torch.Tensor):
            close_embedding_np = close_embedding.cpu().numpy()
        else:
            close_embedding_np = close_embedding
        
        filtered_latent_list.append(close_embedding_np)
        
        # Clean up memory
        few_shot.delete_precomputed_embeddings()
        few_shot.delete_references()
        
        # Force garbage collection
        gc.collect()

    # Compute mean reference embeddings
    mean_ref = torch.from_numpy(np.vstack([np.mean(l, axis=0) for l in filtered_latent_list]))
    
    return mean_ref

def compute_accuracies(reference_embeddings, 
                       embeddings,
                       metric=cosine_similarity,
                       distance_threshold=0.01):
    """
    Compute classification accuracies for each neurotransmitter class.
    
    Args:
        reference_embeddings: Reference embeddings for each class
        embeddings: Test embeddings to classify
        metric: Distance metric function
        distance_threshold: Threshold for filtering distances
    
    Yields:
        list: Batch scores for each class
    """
    
    l = []
    for e in embeddings:
        l.extend(e)
    embeddings = l
    
    batch_size = int(len(embeddings) / len(neurotransmitters))
    one_hot_neurotransmitters = np.eye(len(neurotransmitters))

    # Ensure embeddings are properly shaped
    if isinstance(embeddings, list):
        embeddings = torch.from_numpy(np.array(embeddings))
    if isinstance(reference_embeddings, list):
        reference_embeddings = torch.from_numpy(np.array(reference_embeddings))
    
    # Flatten embeddings to 2D if they have more dimensions
    embeddings = embeddings.reshape(-1, feat_dim)
    reference_embeddings = reference_embeddings.reshape(-1, feat_dim)

    for n, i in tqdm(enumerate(range(0, len(embeddings), batch_size)), 
                     desc='Computing accuracies'):
        batch = embeddings[i:i+batch_size]

        # Compute similarity matrix
        similarity_matrix = metric(reference_embeddings, batch)
        # Normalize similarity matrix
        sim_min, sim_max = np.min(similarity_matrix), np.max(similarity_matrix)
        if sim_max > sim_min:  # Avoid division by zero
            similarity_matrix_normalized = (similarity_matrix - sim_min) / (sim_max - sim_min)
        else:
            similarity_matrix_normalized = similarity_matrix
        
        # Apply threshold
        from .analysis_utils import get_threshold
        threshold = get_threshold(similarity_matrix_normalized, 0.99)
        similarity_matrix_filtered = np.where(
            similarity_matrix_normalized <= threshold, 
            similarity_matrix_normalized, 
            0
        )

        batch_score_list = []
        rejected_count = 0
        
        # Get the actual batch size (might be smaller for the last batch)
        actual_batch_size = batch.shape[0]
        
        for k in range(actual_batch_size):
            column = similarity_matrix_filtered[:, k]
            
            if np.sum(column) == 0:
                rejected_count += 1
                # Could assign random class or skip - here we skip
                continue
            else:
                # Find best matching class
                patch_wise_distances_filtered = np.where(column == 0, 1, column)
                predicted_class_idx = np.argmin(patch_wise_distances_filtered)
                output_class = one_hot_neurotransmitters[predicted_class_idx]
                
                # Ground truth
                ground_truth = one_hot_neurotransmitters[n]
                
                # Compute accuracy score
                score = np.sum(output_class * ground_truth)
                batch_score_list.append(score)
                
        if rejected_count > 0:
            print(f"Warning: {rejected_count} embeddings rejected due to threshold in batch {n}")
            
        yield batch_score_list

def main():
    """Main execution function"""
    
    # Load or compute reference embeddings
    try:
        mean_refs = compute_ref_embeddings(
            saved_ref_embeddings=True, 
            embs_path=os.path.join(embeddings_path, 'small_mean_ref_518_Aug=False_k=10.pt')
        )
    except Exception as e:
        print(f"Error loading reference embeddings: {e}")
        print("Computing new reference embeddings...")
        mean_refs = compute_ref_embeddings(saved_ref_embeddings=False)
    
    # Load test embeddings
    try:
        new_embeddings = torch.load(
            os.path.join(embeddings_path, 'small_dataset_embs_518.pt'),
            map_location=device
        )
    except FileNotFoundError:
        print("Error: Could not load test embeddings file 'small_dataset_embs_518.pt'")
        return
    
    # Create accuracy generator
    accuracy_generator = compute_accuracies(
        reference_embeddings=mean_refs,
        embeddings=new_embeddings,
        metric=euclidean_distances,
        distance_threshold=0.01
    )
    
    # Collect results
    score_list = []
    try:
        for _ in range(len(neurotransmitters)):
            batch_scores = next(accuracy_generator)
            score_list.append(batch_scores)
    except StopIteration:
        print("Warning: Not enough batches generated")
    
    # Calculate accuracies
    accuracies = []
    for scores in score_list:
        if len(scores) > 0:
            accuracies.append(np.mean(scores) * 100)
        else:
            accuracies.append(0.0)  # No valid predictions
    
    # Plotting
    plt.figure(figsize=(12, 7), dpi=200)
    bars = plt.bar(neurotransmitters, accuracies)
    plt.xlabel('Neurotransmitter Classes')
    plt.ylabel('Mean Classification Accuracy (%)')
    plt.title(f'Classification Accuracy by Neurotransmitter Class ({model_size} DINOv2) - Threshold = 0.99')
    
    # Add reference lines
    avg_accuracy = np.mean(accuracies)
    random_accuracy = 100 / len(neurotransmitters)
    
    plt.axhline(avg_accuracy, color='r', linestyle='--', 
                label=f'Average: {avg_accuracy:.1f}%')
    plt.axhline(random_accuracy, color='b', linestyle='--', 
                label=f'Random: {random_accuracy:.1f}%')
    
    plt.legend()
    plt.ylim([0, 105])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Print results
    print("\nClassification Results:")
    print("-" * 40)
    for neuro, acc in zip(neurotransmitters, accuracies):
        print(f"{neuro:15s}: {acc:6.2f}%")
    print("-" * 40)
    print(f"{'Average':15s}: {avg_accuracy:6.2f}%")
    print(f"{'Random baseline':15s}: {random_accuracy:6.2f}%")
    
    plt.show()
    
    return accuracies

if __name__ == '__main__':
    main()