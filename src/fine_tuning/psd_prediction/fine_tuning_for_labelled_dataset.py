import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import f1_score, jaccard_score
import torchvision
import torchvision.transforms as Trans
from pytorch_metric_learning.losses import NTXentLoss
from skimage import measure, morphology

from src.setup import device, model_weights_path, resize_size, feat_dim, neurotransmitters, model_size
from src.perso_utils import load_image
from src.analysis_utils import resize_hdf_image
from src.fine_tuning.adaptor.AdaptFormer import AdaptMLP
from src.fine_tuning.psd_prediction.mlp_head import Psd_Pred_MLP_Head
from src.fine_tuning.display_results import display_segmentation, confusion_matrix
from .sliding_window_dataset_for_labelled_dataset import get_data_generator



# FIXME: check if the max_sample in the evaluate function isn't doing anything funky



class FineTuner:
    """Fine-tuning class for DINOv2 with adapter layers and MLP head"""
    
    def __init__(self, model_size, device, feat_dim, resize_size, 
                 use_contrastive=False, contrastive_weight=0.1):
        self.model_size = model_size
        self.device = device
        self.feat_dim = feat_dim
        self.resize_size = resize_size
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
        # Create classification model
        self.model = self.initialize_model()
        self.patch_size = self.model[0].patch_size
        
        # Loss functions
        self.regression_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        if use_contrastive:
            self.contrastive_loss_fn = NTXentLoss()
            
        # Define early-stopping criterions
        self.patience = 5
        self.min_delta = 0.01
        self.counter = 0
        self.best_score = -np.inf
        self.should_stop = False
 

    def initialize_model(self):
        model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{self.model_size[0]}14_reg')
        model.to(self.device) # type: ignore

        for param in model.parameters(): # type: ignore
            param.requires_grad = False

        for k in range(len(list(model.blocks))): # type: ignore

            mlp = model.blocks[k].mlp# type: ignore
            in_dim = model.blocks[k].norm2.normalized_shape[0]# type: ignore
            mid_dim = int(model.blocks[k].norm2.normalized_shape[0]/10) #TODO: important parameter # type: ignore
            
            adapter = AdaptMLP(self.device,
                            mlp, 
                            in_dim, 
                            mid_dim, 
                            dropout=0.0, 
                            s=0.1)

            model.blocks[k].mlp = adapter# type: ignore

        detection_head = Psd_Pred_MLP_Head(device=device, 
                                        nb_outputs=1, 
                                        feat_dim=feat_dim)

        augmented_model = nn.Sequential(model, detection_head) # type: ignore
        augmented_model.eval()

        trainable_params = [p for p in augmented_model.parameters() if p.requires_grad]
        params = sum([np.prod(p.size()) for p in trainable_params])
        frozen_params_list = [p for p in augmented_model.parameters() if not p.requires_grad]
        frozen_params = sum([np.prod(p.size()) for p in frozen_params_list])
        total_params = params + frozen_params
        print(f'Proportion of trainable parameters when head frozen: {params / total_params * 100:.2f}%')

        return augmented_model.to(device)

    def setup_optimizer(self, training_index, learning_rate=3e-4):
        """Setup optimizer based on which component to train"""
        if training_index == 1:  # Train MLP head
            return self._setup_mlp_optimizer(learning_rate)
        elif training_index == 0:  # Train DINOv2 adapter layers
            return self._setup_adapter_optimizer(learning_rate)
        else:
            raise ValueError("training_index must be 0 (adapter) or 1 (MLP)")
    
    def _setup_mlp_optimizer(self, learning_rate):
        """Setup optimizer for MLP head training"""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze MLP head
        for param in self.model[1].parameters():
            param.requires_grad = True
        
        print("Training MLP Head")
        return torch.optim.Adam(self.model[1].parameters(), lr=learning_rate)
    
    def _setup_adapter_optimizer(self, learning_rate):
        """Setup optimizer for adapter layer training"""
        trainable_params = []
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze adapter layers
        for block in self.model[0].blocks:
            mlp = block.mlp
            
            for param in mlp.down_proj.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            
            for param in mlp.up_proj.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        
        print(f"Training DINOv2 adapter layers - {len(trainable_params)} parameters")
        return torch.optim.Adam(trainable_params, lr=learning_rate)
    
    def extract_patch_features(self, training_index, patches, ground_truth):
        """Extract features and targets from patches - Fixed version"""
        H_size, W_size = ground_truth.shape
        H_patch, W_patch = H_size // self.patch_size, W_size // self.patch_size
        
        all_embeddings = []
        all_targets = []
        
        # Use torch.no_grad() to avoid building computation graph unnecessarily
        
        for n, patch in enumerate(patches):
            patch = patch.to(self.device).float()
            nb_patches_h, nb_patches_w = patch.shape[-2] // self.patch_size, patch.shape[-1] // self.patch_size
            
            # Extract embeddings
            if training_index == 1:
                with torch.no_grad():
                    embeddings = self.model[0].forward_features(patch)["x_norm_patchtokens"]
            else:
                embeddings = self.model[0].forward_features(patch)["x_norm_patchtokens"]
                
            embeddings = embeddings.reshape(nb_patches_h, nb_patches_w, self.feat_dim)
            central_embedding = embeddings[nb_patches_h//2, nb_patches_w//2]
            all_embeddings.append(central_embedding)
            
            # Calculate target
            h_coord = (n // W_patch) * self.patch_size
            w_coord = (n % W_patch) * self.patch_size
            gt_patch = ground_truth[h_coord:h_coord + self.patch_size, 
                                    w_coord:w_coord + self.patch_size]
            target = gt_patch.mean()
            all_targets.append(target)
        
        # Stack tensors - they will automatically require gradients if model is in train mode
        all_embeddings_tensor = torch.stack(all_embeddings)
        all_targets_tensor = torch.stack(all_targets)
        
        return all_embeddings_tensor, all_targets_tensor
    
    def compute_contrastive_loss(self, embeddings, targets):
        """Compute contrastive loss separately"""
        if not self.use_contrastive:
            return torch.tensor(0.0, device=self.device)
        
        # Convert targets to discrete labels for contrastive learning
        discrete_targets = (targets > 0.5).long()
        return self.contrastive_loss_fn(embeddings, discrete_targets)
    
    def train_epoch(self, train_set, optimizer, training_index, batch_size=32):
        """Fixed training epoch function"""
        self.model.train()
        epoch_losses = []
        regression_losses = []
        contrastive_losses = []
        
        for file, patches, ground_truth in train_set:
            ground_truth = ground_truth.to(self.device).float()
            
            # Extract features once per image
            embeddings, targets = self.extract_patch_features(training_index, patches, ground_truth)
            
            if len(embeddings) == 0:
                continue
            
            # Process in batches to avoid memory issues
            num_patches = len(embeddings)
            for i in range(0, num_patches, batch_size):
                optimizer.zero_grad()  # Clear gradients for each batch
                total_loss = torch.tensor(0.0, device=self.device)
                
                end_idx = min(i + batch_size, num_patches)
                if training_index == 1:
                    batch_embeddings = embeddings[i:end_idx].detach().requires_grad_(True)
                else:
                    batch_embeddings = embeddings[i:end_idx].clone().detach().requires_grad_(True)
                    
                batch_targets = targets[i:end_idx]
                
                # Contrastive loss (if enabled)
                if self.use_contrastive and training_index == 0:
                    contrastive_loss = self.compute_contrastive_loss(batch_embeddings, batch_targets)
                    total_loss = total_loss + self.contrastive_weight * contrastive_loss
                    contrastive_losses.append(contrastive_loss.item())
                    print(f"Contrastive Loss: {contrastive_loss.item():.4f}")
                
                # Regression loss
                predictions = self.model[1](batch_embeddings).squeeze()
                
                '''
                plt.imshow(predictions.unsqueeze(0).detach().cpu().numpy())
                plt.show()  
                plt.imshow(batch_targets.unsqueeze(0).detach().cpu().numpy())
                plt.show()
                '''
                
                # Handle single prediction case
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if batch_targets.dim() == 0:
                    batch_targets = batch_targets.unsqueeze(0)
                
                # Calculate regression loss (keep everything on same device)
                regression_loss = self.regression_loss_fn(predictions, batch_targets)
                
                print(f"Regression Loss: {regression_loss.item():.4f}")

                
                total_loss = total_loss + regression_loss
                regression_losses.append(regression_loss.item())
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
        
        # Print detailed loss information
        avg_total_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_regression_loss = np.mean(regression_losses) if regression_losses else 0.0
        avg_contrastive_loss = np.mean(contrastive_losses) if contrastive_losses else 0.0
        
        if self.use_contrastive and training_index == 0:
            print(f"  Regression Loss: {avg_regression_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}")
        
        return avg_total_loss
    
    def evaluate(self, test_set, use_umap=False, max_samples=None, metric='jaccard', epoch=None, use_display_segmentation=False):
        """Fixed evaluation function with consistent metrics"""
        self.model.eval()
        scores = []
        umap_data = {'embeddings': [], 'labels': []} if use_umap else None
        
        samples_processed = 0
        with torch.no_grad():
            for file, patches, ground_truth in test_set:
                if max_samples and samples_processed >= max_samples:
                    break
                
                ground_truth = ground_truth.to(self.device).float()
                prediction = self._predict_image(patches, ground_truth, umap_data)
                
                # Post-process prediction - USE the result this time
                prediction_bool = prediction.astype(bool)
                inverted_prediction = ~prediction_bool
                processed_prediction = self._keep_largest_continuous_mask(inverted_prediction)
                
                # Use processed prediction for scoring
                final_prediction = (~processed_prediction.astype(bool)).astype(int)
                
                # Calculate score with chosen metric
                ground_truth_np = ground_truth.cpu().numpy().astype(int)
                
                if metric == 'jaccard':
                    score = jaccard_score(ground_truth_np, prediction, # FIXME: use final_prediction
                                        zero_division=0, average='micro')
                elif metric == 'f1':
                    score = f1_score(ground_truth_np.flatten(), prediction.flatten(), 
                                   average='binary', zero_division=0)
                else:
                    raise ValueError("metric must be 'jaccard' or 'f1'")
                
                scores.append(score)
                
                # Optional: Display results (uncomment if needed)
                if use_display_segmentation:
                    display_segmentation(final_prediction, file, score)
                    display_segmentation(ground_truth_np, file, score)
                
                samples_processed += 1
        
        avg_score = np.mean(scores) if scores else 0.0
        
        # UMAP visualization
        if use_umap and umap_data['embeddings']:
            self._create_umap_visualization(umap_data, avg_score)
        
        return avg_score
    
    def _predict_image(self, patches, ground_truth, umap_data=None):
        """Fixed prediction function"""
        H_size, W_size = ground_truth.shape
        H_patch, W_patch = H_size // self.patch_size, W_size // self.patch_size
        prediction = torch.zeros(ground_truth.shape, device=self.device)
        
        for n, patch in enumerate(patches):
            patch = patch.to(self.device).float()
            nb_patches_h, nb_patches_w = patch.shape[-2] // self.patch_size, patch.shape[-1] // self.patch_size
            
            # Extract embedding
            embeddings = self.model[0].forward_features(patch)["x_norm_patchtokens"]
            embeddings = embeddings.reshape(nb_patches_h, nb_patches_w, self.feat_dim)
            central_embedding = embeddings[nb_patches_h//2, nb_patches_w//2].unsqueeze(0)
            
            # Predict
            output = self.model[1](central_embedding)
            output_value = (torch.sigmoid(output) > 0.5).float()
            
            # Collect UMAP data
            if umap_data is not None:
                umap_data['embeddings'].append(central_embedding.cpu().numpy().flatten())
                umap_data['labels'].append(output_value.cpu().numpy())
            
            # Update prediction map
            h_coord = (n // W_patch) * self.patch_size
            w_coord = (n % W_patch) * self.patch_size
            prediction[h_coord:h_coord + self.patch_size, 
                      w_coord:w_coord + self.patch_size] = output_value

        return prediction.cpu().numpy()
    
    def _keep_largest_continuous_mask(self, mask):
        """Keep only the largest continuous region in the mask"""
        if not np.any(mask):  # Handle empty mask
            return mask.astype(np.uint8)
            
        labelled = measure.label(mask)
        rp = measure.regionprops(labelled)
        
        if not rp:
            return mask.astype(np.uint8)
        
        # Get size of largest cluster
        max_size = max(region.area for region in rp)
        
        # Remove everything smaller than largest
        out = morphology.remove_small_objects(mask, min_size=max_size-1)
        return out.astype(np.uint8)
    
    def _create_umap_visualization(self, umap_data, test_score):
        """Fixed UMAP visualization"""
        if len(umap_data['embeddings']) == 0:
            print("No embeddings available for UMAP visualization")
            return
            
        reducer = umap.UMAP(random_state=42)
        embeddings_array = np.array(umap_data['embeddings'])
        
        # Handle single embedding case
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
            
        embeddings_2d = reducer.fit_transform(embeddings_array)
        
        labels = np.array(umap_data['labels']).flatten()
        binary_labels = (labels > 0.5).astype(int)
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']  # Rest and PSD
        
        for label, color, name in zip([0, 1], colors, ['Rest', 'PSD']):
            mask = binary_labels == label
            if np.sum(mask) > 0:
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=color, s=1, label=name, alpha=0.6)
        
        plt.legend()
        plt.title(f'UMAP Visualization - Test Score: {test_score:.4f}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.show()


    def _early_stopping(self, test_accuracy, epoch):
        if test_accuracy > (self.best_score + self.min_delta):
            self.best_score = test_accuracy
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch} with test accuracy {test_accuracy} and patience {self.patience}")
                return True
        return False
    
    def reset_early_stopping(self):
        self.counter = 0
        self.best_score = -np.inf
        self.should_stop = False


    def train_component(self, training_index, epochs, train_set, test_set, 
                       batch_size=32, eval_frequency=1, metric='jaccard',
                       use_early_stopping=True):
        """Fixed training component function"""
        component_names = ['DINOv2 Adapter', 'MLP Head']
        print(f"Training {component_names[training_index]}...")
        
        if use_early_stopping:
            self.reset_early_stopping()
        
        optimizer = self.setup_optimizer(training_index)
        eval_frequency = eval_frequency or max(1, epochs // 10)
        
        train_losses = []
        test_scores = []
        
        for epoch in tqdm(range(epochs), desc=f'Training {component_names[training_index]}'):
            # Training
            train_loss = self.train_epoch(train_set, optimizer, training_index, batch_size)
            train_losses.append(train_loss)
            
            # Evaluation
            if epoch % eval_frequency == 0 or epoch == epochs - 1:
                # Full evaluation on last epoch, limited during training
                max_samples = None if epoch == epochs - 1 else 50
                test_score = self.evaluate(test_set, max_samples=max_samples, metric=metric, epoch=epoch)
                test_scores.append(test_score)
                
                metric_name = 'IoU' if metric == 'jaccard' else 'F1'
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, {metric_name}: {test_score:.4f}")
                
                if use_early_stopping and self._early_stopping(test_score, epoch):
                    print('Early stopping triggered at epoch {epoch}'.format(epoch=epoch))
                    save_checkpoint(model=self.model, 
                                    test_accuracy_lists=test_scores, 
                                    iteration=epoch, 
                                    filename=f"checkpoint_iter_{epoch}.pth")
                    break
                
            else:
                # Reuse last test score for non-evaluation epochs
                test_scores.append(test_scores[-1] if test_scores else 0.0)
        
        return train_losses, test_scores


def save_checkpoint(model, test_accuracy_lists, iteration, filename=None):
    """Save model checkpoint with training statistics"""
    if filename is None:
        filename = f"checkpoint_iter_{iteration}.pth"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'test_accuracy_lists': test_accuracy_lists,
        'iteration': iteration
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def plot_training_results(loss_lists, test_lists, metric='jaccard'):
    """Fixed plotting function"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
    
    # Training loss plot
    if loss_lists[0]:  # DINOv2
        x_dino = list(range(1, len(loss_lists[0]) + 1))
        ax1.plot(x_dino, loss_lists[0], label='DINOv2 Train Loss', 
                color='blue', marker='o', linewidth=2)
    
    if loss_lists[1]:  # MLP
        x_mlp = list(range(1, len(loss_lists[1]) + 1))
        ax1.plot(x_mlp, loss_lists[1], label='MLP Train Loss', 
                color='cyan', marker='s', linewidth=2)
    
    ax1.set_ylabel('Train Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_title('Training Loss Over Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test score plot
    metric_name = 'IoU' if metric == 'jaccard' else 'F1'
    
    if test_lists[0]:
        x_dino = list(range(1, len(test_lists[0]) + 1))
        ax2.plot(x_dino, test_lists[0], label=f'DINOv2 Test {metric_name}', 
                color='red', marker='o', linewidth=2)
    
    if test_lists[1]:
        x_mlp = list(range(1, len(test_lists[1]) + 1))
        ax2.plot(x_mlp, test_lists[1], label=f'MLP Test {metric_name}', 
                color='orange', marker='s', linewidth=2)
    
    ax2.set_ylabel(f'Test {metric_name} Score')
    ax2.set_xlabel('Iterations')
    ax2.set_title(f'Test {metric_name} Score Over Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_final_results(loss_lists, test_lists, metric='jaccard'):
    """Print comprehensive final results"""
    print(f"\n{'='*50}")
    print(f"FINAL TRAINING RESULTS")
    print(f"{'='*50}")
    
    component_names = ['DINOv2 Adapter', 'MLP Head']
    metric_name = 'IoU' if metric == 'jaccard' else 'F1'
    
    for i, name in enumerate(component_names):
        if loss_lists[i] and test_lists[i]:
            final_loss = loss_lists[i][-1]
            final_score = test_lists[i][-1]
            print(f"{name} - Train Loss: {final_loss:.4f}, Test {metric_name}: {final_score:.4f}")
            
            # Calculate improvement
            if len(test_lists[i]) > 1:
                improvement = test_lists[i][-1] - test_lists[i][0]
                print(f"{name} {metric_name} Improvement: {improvement:+.4f}")


def run_fine_tuning(nb_iterations=2, nb_epochs_per_iteration=10, 
                   nb_best_patches=100, padding_size=2, use_contrastive=False,
                   contrastive_weight=0.1, metric='jaccard'):
    """Fixed main fine-tuning function"""
    print("Initializing fine-tuner...")
    
    # Initialize fine-tuner with contrastive learning option
    fine_tuner = FineTuner('small', device, feat_dim, resize_size,
                          use_contrastive=use_contrastive, 
                          contrastive_weight=contrastive_weight)
    
    # Load datasets
    print("Loading datasets...")
    train_set = list(get_data_generator(
        split='training', 
        resize_size=resize_size, 
        padding_size=padding_size, 
        test_proportion=0, # TODO: change
        seed=42)
    )
    test_set = list(get_data_generator(
        split='test',
        resize_size=resize_size, 
        padding_size=padding_size, 
        test_proportion=1, # TODO: change
        seed=42)
    )
    
    # Training schedule: alternating MLP Head (1) and DINOv2 Adapter (0)
    training_schedule = []
    for _ in range(nb_iterations):
        training_schedule.extend([1, 0])  # MLP first, then adapter
    
    # Track results
    loss_lists = [[], []]  # [DINOv2, MLP Head]
    test_lists = [[], []]
    
    # Training loop
    iteration = 1
    for i, training_index in enumerate(training_schedule):
        print(f"\n{'='*80}")
        component_name = 'MLP Head' if training_index == 1 else 'DINOv2 Adapter'
        print(f"Iteration {iteration} - Training {component_name}")
        print(f"{'='*80}")
        
        # Train component
        train_losses, test_scores = fine_tuner.train_component(
            training_index=training_index,
            epochs=nb_epochs_per_iteration,
            train_set=train_set,
            test_set=test_set,
            batch_size=32,
            metric=metric
        )
        
        # Store results
        final_loss = train_losses[-1] if train_losses else 0.0
        final_score = test_scores[-1] if test_scores else 0.0
        
        loss_lists[training_index].append(final_loss)
        test_lists[training_index].append(final_score)
        
        metric_name = 'IoU' if metric == 'jaccard' else 'F1'
        print(f'{component_name} | Iteration: {iteration} | Loss: {final_loss:.4f} | {metric_name}: {final_score:.4f}')
        
        # Save checkpoint every full iteration (after both components)
        if i % 2 == 1:  # After both components trained
            #save_checkpoint(fine_tuner.model, loss_lists, test_lists, iteration)
            iteration += 1
    
    # Final evaluation with UMAP
    print("\nRunning final evaluation with UMAP...")
    final_score = fine_tuner.evaluate(test_set, use_umap=True, metric=metric, use_display_segmentation=True)
    metric_name = 'IoU' if metric == 'jaccard' else 'F1'
    print(f"Final Test {metric_name} Score: {final_score:.4f}")
    
    # Save final model
    #save_checkpoint(fine_tuner.model, loss_lists, test_lists, "final")
    
    # Plot and print results
    plot_training_results(loss_lists, test_lists, metric)
    print_final_results(loss_lists, test_lists, metric)
    
    return loss_lists, test_lists, fine_tuner


if __name__ == '__main__':
    print("Starting fine-tuning process...")
    
    # Run fine-tuning with optimized parameters
    loss_lists, test_lists, trained_model = run_fine_tuning(
        nb_iterations=5,  # Reduced for testing
        nb_epochs_per_iteration=20,  # Reduced for faster training
        padding_size=3,
        use_contrastive=True,  # Disable contrastive learning by default
        contrastive_weight=0.1,
        metric='jaccard'  # Use Jaccard (IoU) by default
    )
    
    print("\nFine-tuning completed successfully!")