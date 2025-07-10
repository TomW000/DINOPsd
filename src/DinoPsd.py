from .setup import umap, torch, tqdm, np, px, PCA, KMeans, h5py, os, ceil, floor
from .setup import device, model, feat_dim, resize_size, dataset_path, neurotransmitters
from .DinoPsd_utils import resizeLongestSide, mirror_border, crop_data_with_overlap


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


image_size = 1120


class DinoPsd_pipeline:
    """A pipeline for computing and managing DINOSim.

    This class handles the computation of DINOSim, using DINOv2 embeddings, manages reference
    vectors, and computes similarity distances. It supports processing large images through
    a sliding window approach with overlap.

    Args:
        model: The DINOv2 model to use for computing embeddings
        model_patch_size (int): Size of patches used by the DINOv2 model
        device: The torch device to run computations on (CPU or GPU)
        img_preprocessing: Function to preprocess images before computing embeddings
        feat_dim (int): Number of features of the embeddings
        dino_image_size (int, optional): Size of the image to be fed to DINOv2. Images will be resized to that size before computing them with DINOv2. Defaults to 518.
    """

    def __init__(
        self,
        model,
        model_patch_size,
        device,
        img_preprocessing,
        feat_dim,
        dino_image_size=518,
    ):
        self.model = model
        self.dino_image_size = dino_image_size
        self.patch_h = self.patch_w = self.embedding_size = (
            dino_image_size // model_patch_size
        )
        self.img_preprocessing = img_preprocessing
        self.device = device
        self.feat_dim = feat_dim

        self.reference_color = torch.zeros(feat_dim, device=device)
        self.reference_emb = torch.zeros(
            (self.embedding_size * self.embedding_size, feat_dim),
            device=device,
        )
        self.exist_reference = False

        self.embeddings = np.array([])
        self.emb_precomputed = False
        self.original_size = []
        self.overlap = (0, 0) #(0.5, 0.5)
        self.padding = (0, 0)
        self.crop_shape = (518, 518, 1)
        self.resized_ds_size, self.resize_pad_ds_size = [], []

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def pre_compute_embeddings(
        self,
        dataset,
        overlap=(0, 0),
        padding=(0, 0),
        crop_shape=(518, 518, 1),
        verbose=True,
        batch_size=60,
    ):
        """Pre-compute DINO embeddings for the entire dataset.

        The dataset is processed in crops with optional overlap. Large images are handled
        through a sliding window approach, and small images are resized.

        Args:
            dataset: Input image dataset with shape (batch, height, width, channels)
            overlap (tuple, optional): Overlap fraction (y, x) between crops. Defaults to (0.5, 0.5).
            padding (tuple, optional): Padding size (y, x) for crops. Defaults to (0, 0).
            crop_shape (tuple, optional): Size of crops (height, width, channels). Defaults to (512, 512, 1).
            verbose (bool, optional): Whether to show progress bar. Defaults to True.
            batch_size (int, optional): Batch size for processing. Defaults to 1.
        """
        print("Precomputing embeddings")
        self.original_size = dataset.shape
        self.overlap = overlap
        self.padding = padding
        self.crop_shape = crop_shape
        b, h, w, c = dataset.shape
        self.resized_ds_size, self.resize_pad_ds_size = [], []

        # if both image resolutions are smaller than the patch size,
        # resize until the largest side fits the patch size
        if h < crop_shape[0] and w < crop_shape[0]:
            dataset = np.array(
                [
                    resizeLongestSide(np_image, crop_shape[0])
                    for np_image in dataset
                ]
            )
            if len(dataset.shape) == 3:
                dataset = dataset[..., np.newaxis]
            self.resized_ds_size = dataset.shape

        # yet if one of the image resolutions is smaller than the patch size,
        # add mirror padding until smaller side fits the patch size
        if (
            dataset.shape[1] % crop_shape[0] != 0
            or dataset.shape[2] % crop_shape[1] != 0
        ):
            desired_h, desired_w = (
                np.ceil(dataset.shape[1] / crop_shape[0]) * crop_shape[0],
                np.ceil(dataset.shape[2] / crop_shape[1]) * crop_shape[1],
            )
            dataset = np.array(
                [
                    mirror_border(
                        np_image, sizeH=int(desired_h), sizeW=int(desired_w)
                    )
                    for np_image in dataset
                ]
            )
            self.resize_pad_ds_size = dataset.shape

        # needed format: b,h,w,c
        windows = crop_data_with_overlap(
            dataset,
            crop_shape=crop_shape,
            overlap=overlap,
            padding=padding,
            verbose=False,
        )
        windows = torch.tensor(windows, device=self.device)
        windows = self._quantile_normalization_fixed(windows.float()) #FIXME: RuntimeError: quantile() input tensor is too large
        prep_windows = self.img_preprocessing(windows)

        self.delete_precomputed_embeddings()
        self.embeddings = torch.zeros(
            (len(windows), self.patch_h, self.patch_w, self.feat_dim),
            device=self.device,
        )

        following_f = tqdm if verbose else lambda aux: aux
        for i in following_f(range(0, len(prep_windows), batch_size)):
            batch = prep_windows[i : i + batch_size]
            b, h, w, c = batch.shape  # b,h,w,c
            crop_h, crop_w, _ = crop_shape
            overlap = (
                overlap[0] if w > crop_w else 0,
                overlap[1] if h > crop_h else 0,
            )

            with torch.no_grad():
                if self.model is None:
                    raise ValueError("Model is not initialized")
                encoded_window = self.model.forward_features(batch)[
                    "x_norm_patchtokens"
                ]
            self.embeddings[i : i + batch_size] = encoded_window.reshape(
                encoded_window.shape[0],
                self.patch_h,
                self.patch_w,
                self.feat_dim,
            )  # use all dims

        self.emb_precomputed = True
        
    def get_embeddings(self, reshape=True):
        if reshape:
            return self.embeddings.reshape(-1, feat_dim)
        else:
            return self.embeddings

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _quantile_normalization(
        self,
        tensor,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        """Normalize tensor values between quantile bounds.

        Args:
            tensor (torch.Tensor): Input tensor to normalize
            lower_quantile (float): Lower quantile bound (0-1)
            upper_quantile (float): Upper quantile bound (0-1)

        Returns:
            torch.Tensor: Normalized tensor with values between 0 and 1
        """
        flat_tensor = tensor.flatten()
        bounds = torch.quantile( #FIXME: RuntimeError: quantile() input tensor is too large

            flat_tensor,
            torch.tensor(
                [lower_quantile, upper_quantile], device=tensor.device
            ),
        )
        lower_bound, upper_bound = bounds[0], bounds[1]

        clipped_tensor = torch.clamp(tensor, lower_bound, upper_bound)
        normalized_tensor = (clipped_tensor - lower_bound) / (
            upper_bound - lower_bound + 1e-8
        )
        return normalized_tensor

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _quantile_normalization_fixed(
        self,
        tensor,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        flat_tensor = tensor.flatten()
        n = flat_tensor.numel()

        k_lower = max(1, int(lower_quantile * n))
        k_upper = min(n, int(upper_quantile * n))
        
        lower_bound = torch.kthvalue(flat_tensor, k_lower).values
        upper_bound = torch.kthvalue(flat_tensor, k_upper).values

        clipped_tensor = torch.clamp(tensor, lower_bound, upper_bound)
        normalized_tensor = (clipped_tensor - lower_bound) / (
            upper_bound - lower_bound + 1e-8
        )
        return normalized_tensor

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def delete_precomputed_embeddings(
        self,
    ):
        """Delete precomputed embeddings to free memory."""
        del self.embeddings
        self.embeddings = np.array([])
        self.emb_precomputed = False
        torch.cuda.empty_cache()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def delete_references(
        self,
    ):
        """Delete reference vectors to free memory."""
        del self.reference_color, self.reference_emb, self.exist_reference
        self.reference_color = torch.zeros(self.feat_dim, device=self.device)
        self.reference_emb = torch.zeros(
            (self.embedding_size * self.embedding_size, self.feat_dim),
            device=self.device,
        )
        self.exist_reference = False
        torch.cuda.empty_cache()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_reference_vector(self, list_coords, filter=None):
        """Set reference vectors from a list of coordinates in the original image space.

        Computes mean embeddings from the specified coordinates to use as reference vectors
        for similarity computation.

        Args:
            list_coords: List of tuples (batch_idx, z, x, y) specifying reference points
            filter: Optional filter to apply to the generated pseudolabels
        """
        self.delete_references()
        if len(self.resize_pad_ds_size) > 0:
            b, h, w, c = self.resize_pad_ds_size
            if len(self.resized_ds_size) > 0:
                original_resized_h, original_resized_w = self.resized_ds_size[
                    1:3
                ]
            else:
                original_resized_h, original_resized_w = self.original_size[
                    1:3
                ]
        elif len(self.resized_ds_size) > 0:
            b, h, w, c = self.resized_ds_size
            original_resized_h, original_resized_w = h, w
        else:
            b, h, w, c = self.original_size
            original_resized_h, original_resized_w = h, w

        n_windows_h = int(np.ceil(h / self.crop_shape[0]))
        n_windows_w = int(np.ceil(w / self.crop_shape[1]))

        # Calculate actual scaling factors
        scale_x = original_resized_w / self.original_size[2]
        scale_y = original_resized_h / self.original_size[1]

        # Calculate padding
        pad_left = (w - original_resized_w) / 2
        pad_top = (h - original_resized_h) / 2

        list_ref_colors, list_ref_embeddings = [], []
        for n, x, y in list_coords:
            # Apply scaling and padding to coordinates
            x_transformed = x * scale_x + pad_left
            y_transformed = y * scale_y + pad_top

            # Calculate crop index and relative position within crop
            n_crop = int(
                np.floor(x_transformed / self.crop_shape[1])
                + np.floor(y_transformed / self.crop_shape[0]) * n_windows_w
            )
            x_coord = (x_transformed % self.crop_shape[1]) / self.crop_shape[1]
            y_coord = (y_transformed % self.crop_shape[0]) / self.crop_shape[0]

            emb_id = int(n_crop + n * n_windows_h * n_windows_w)

            # Validate embedding index
            if emb_id >= len(self.embeddings):
                raise ValueError(
                    f"Invalid embedding index {emb_id} for coordinates ({n}, {x}, {y})"
                )

            x_coord = min(
                round(x_coord * self.embedding_size), self.embedding_size - 1
            )
            y_coord = min(
                round(y_coord * self.embedding_size), self.embedding_size - 1
            )

            list_ref_colors.append(self.embeddings[emb_id][y_coord, x_coord])
            list_ref_embeddings.append(self.embeddings[emb_id])

        list_ref_colors, list_ref_embeddings = torch.stack(
            list_ref_colors
        ), torch.stack(list_ref_embeddings)
        assert (
            len(list_ref_colors) > 0
        ), "No binary objects found in given masks"

        self.reference_color = torch.mean(list_ref_colors, dim=0)
        self.reference_emb = list_ref_embeddings
        self.generate_pseudolabels(filter)
        self.exist_reference = True

    def get_reference_embedding(self):
        return self.reference_color
        
    def get_reference_patches(self):
        return self.reference_emb

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def generate_pseudolabels(self, filter=None):
        """Generate pseudolabels using reference embeddings.
        
        Args:
            filter: Optional filter to apply to the generated distances
        """
        reference_embeddings = self.reference_emb.view(
            -1, self.reference_emb.shape[-1]
        )
        distances = torch.cdist(
            reference_embeddings, self.reference_color[None], p=2
        )

        if filter != None:
            distances = distances.view(
                (
                    self.reference_emb.shape[0],
                    1,
                    int(self.embedding_size),
                    int(self.embedding_size),
                )
            )
            distances = filter(distances)

        # normalize per image
        distances = self.quantile_normalization(distances)

        self.reference_pred_labels = distances.view(-1, 1)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def quantile_normalization(
        self, tensor, lower_quantile=0.01, upper_quantile=0.99
    ):
        """Normalize tensor values between quantile bounds.
        
        Args:
            tensor (torch.Tensor): Input tensor to normalize
            lower_quantile (float): Lower quantile bound (0-1)
            upper_quantile (float): Upper quantile bound (0-1)
            
        Returns:
            torch.Tensor: Normalized tensor with values between lower and upper bounds
        """
        sorted_tensor, _ = tensor.flatten().sort()
        lower_bound = sorted_tensor[
            int(lower_quantile * (len(sorted_tensor) - 1))
        ]
        upper_bound = sorted_tensor[
            int(upper_quantile * (len(sorted_tensor) - 1))
        ]

        clipped_tensor = torch.clamp(tensor, lower_bound, upper_bound)
        return (clipped_tensor - lower_bound) / (
            upper_bound - lower_bound + 1e-8
        )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_d_closest_elements_gen(self, embeddings, reference_emb, d=0.5, normalize_distances=True, verbose=False, chunk_size=100):
        """
        Generator version: yields (embedding, label) pairs within threshold distance `d`.
        """
        if verbose:
            print("Finding elements close to reference vectors...")

        if type(embeddings) == np.ndarray:
            all_embeddings = torch.from_numpy(embeddings)
        elif isinstance(embeddings, list):
            all_embeddings = torch.from_numpy(np.array(embeddings)).reshape(-1, self.feat_dim)
        else:    
            all_embeddings = embeddings.reshape(-1, self.feat_dim)
            
        ref_embeddings = reference_emb.reshape(-1, self.feat_dim)
        
        if type(ref_embeddings) == np.ndarray:
            ref_embeddings = torch.from_numpy(ref_embeddings)
        elif isinstance(ref_embeddings, list):
            ref_embeddings = torch.from_numpy(np.array(ref_embeddings))
        else:
            ref_embeddings = ref_embeddings

        # Precompute distances once if needed (rarely)
        if normalize_distances:
            all_dists = []

        emb_labels = np.hstack([[neuro]*int((resize_size/14)**2 * 600) for neuro in neurotransmitters]).reshape(-1, 1)

        num_embeddings = all_embeddings.shape[0]
        for i in range(0, num_embeddings, chunk_size):
            chunk = all_embeddings[i:i + chunk_size]  # [chunk_size, feat_dim]
            distances = torch.cdist(chunk, ref_embeddings, p=2)  # [chunk_size, ref_size]
            min_dists, _ = torch.min(distances, dim=1)

            if normalize_distances:
                min_dists = self.quantile_normalization(min_dists.view(-1, 1)).squeeze()

            # Mask: which distances are within threshold
            mask = min_dists < d
            selected_indices = torch.nonzero(mask).squeeze().cpu().numpy()

            for idx in selected_indices:
                global_idx = i + idx
                yield chunk[idx].cpu(), emb_labels[global_idx]

    def get_d_closest_elements(self, embeddings, reference_emb, d=0.5, normalize_distances=True, verbose=False, chunk_size=100):
        g = self.get_d_closest_elements_gen(embeddings, reference_emb, d=0.5, normalize_distances=True, verbose=False, chunk_size=100)
        dataset = []
        if isinstance(embeddings, list):
            nb_embeddings = len(embeddings)*3600*((resize_size/14)**2)
        else:
            nb_embeddings = embeddings.shape[0]
        for item in tqdm(g, desc="Streaming filtered embeddings"):
            dataset.append(item)
        return dataset

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_k_closest_elements(self, k=10, return_indices=False):
        """
        Get the k elements in the embedding space that are closest to the reference vector.

        Args:
            k (int, optional): Number of nearest neighbors to return. Defaults to 5.
            return_indices (bool, optional): Whether to return the indices of the closest elements. Defaults to False.

        Returns:
            torch.Tensor: Tensor containing the k nearest embeddings to the reference vector
            torch.Tensor (optional): Indices of the k nearest embeddings if return_indices is True
        """
        if not self.exist_reference:
            raise ValueError("Reference vector not set. Call set_reference_vector first.")

        if not self.emb_precomputed:
            raise ValueError("Embeddings not computed. Call pre_compute_embeddings first.")

        # Flatten all embeddings to 2D tensor: (num_embeddings*h*w, feat_dim)
        all_embeddings = self.get_embeddings(reshape=True)
        
        reference_embedding = self.get_reference_embedding()

        # Compute distances between all embeddings and the reference color vector
        distances = torch.cdist(all_embeddings, reference_embedding.unsqueeze(0), p=2)
        #distances = torch.cdist(all_embeddings, ref_embedings, p=2)

        # Get the k smallest distances and their indices
        k_smallest_values, k_smallest_indices = torch.topk(distances.squeeze(), k=k, largest=False)

        # Get the corresponding embeddings
        closest_embeddings = all_embeddings[k_smallest_indices]

        if return_indices:
            # Convert flat indices back to (batch_idx, y, x) coordinates
            batch_size = len(self.embeddings)
            patch_size = self.patch_h * self.patch_w

            batch_indices = k_smallest_indices // patch_size
            remaining_indices = k_smallest_indices % patch_size
            y_indices = remaining_indices // self.patch_w
            x_indices = remaining_indices % self.patch_w

            indices = torch.stack([batch_indices, y_indices, x_indices], dim=1)
            return closest_embeddings, indices, k_smallest_values

        return closest_embeddings

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Alias method for backwards compatibility or alternative naming convention
    def get_closest_elements(self, d=0.5, normalize_distances=True, return_indices=True, verbose=False):
        """Alias for get_elements_close_to_any_reference."""
        return self.get_elements_close_to_any_reference(d, normalize_distances, return_indices, verbose)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def diplay_features(embeddings,
                    labels,
                    include_pca=False,
                    pca_nb_components=10,
                    clustering=True,
                    nb_clusters=6,
                    nb_neighbor=5,
                    min_dist=0.01,
                    nb_components=2,
                    metric='correlation'):
    """Display features using PCA and UMAP for dimensionality reduction.
    
    Args:
        embeddings: Feature embeddings to visualize
        labels: Labels for coloring the points
        include_pca (bool): Whether to apply PCA before UMAP
        pca_nb_components (int): Number of PCA components if PCA is used
        nb_neighbor (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        nb_components (int): Number of UMAP components
        metric (str): Distance metric for UMAP
    """
    if len(embeddings):

        if include_pca:
            pca = PCA(n_components=pca_nb_components)
            features = pca.fit_transform(embeddings)
        else:
            features = embeddings

        reducer = umap.UMAP(
            n_neighbors=nb_neighbor,
            min_dist=min_dist,
            n_components=nb_components,
            metric=metric, 
            random_state=42
            )
        embedding = reducer.fit_transform(features)
        
        if clustering:
            kmeans = KMeans(n_clusters=nb_clusters)
            kmeans.fit_transform(embedding)
            fig = px.scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                color = kmeans.labels_,
                title = f'KMeans={clustering} ({nb_clusters}) - PCA={include_pca} ({pca_nb_components}) - UMAP (n_neighbors={nb_neighbor}, min_dist={min_dist}, n_components={nb_components}, metric={metric})',
                width=1500,
                height=1000
                )

        else:

            fig = px.scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                color = labels,
                title = f'PCA={include_pca} ({pca_nb_components}) - UMAP (n_neighbors={nb_neighbor}, min_dist={min_dist}, n_components={nb_components}, metric={metric})',
                width=1000,
                height=666
            )

        fig.show()
    else:
        print("No features were extracted!")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------