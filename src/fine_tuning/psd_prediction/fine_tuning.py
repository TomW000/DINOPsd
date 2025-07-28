import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import f1_score, jaccard_score
import torchvision
import torchvision.transforms as Trans
from pytorch_metric_learning.losses import NTXentLoss
from skimage import measure, morphology

from src.setup import model, device, model_weights_path, resize_size, feat_dim, neurotransmitters
from src.perso_utils import load_image
from src.analysis_utils import resize_hdf_image
from src.fine_tuning.adaptor.AdaptFormer import model
from src.fine_tuning.psd_prediction.mlp_head import detection_head, classification_head
from src.fine_tuning.display_results import display_segmentation, confusion_matrix
from .sliding_window_dataset import get_data_generator


augmented_model = nn.Sequential(model, detection_head) # type: ignore
augmented_model.eval()
augmented_model.to(device)


trainable_params = [p for p in augmented_model.parameters() if p.requires_grad]
params = sum([np.prod(p.size()) for p in trainable_params])
frozen_params_list = [p for p in augmented_model.parameters() if not p.requires_grad]
frozen_params = sum([np.prod(p.size()) for p in frozen_params_list])
total_params = params + frozen_params
print(f'Proportion of trainable parameters when head frozen: {params / total_params * 100:.2f}%')



class FineTuner:
    """Fine-tuning class for DINOv2 with adapter layers and MLP head"""
    
    def __init__(self, augmented_model, device, feat_dim, resize_size, patch_size=None):
        self.device = device
        self.feat_dim = feat_dim
        self.resize_size = resize_size
        self.patch_size = patch_size or augmented_model[0].patch_size
        
        # Create classification model
        self.classification_head = classification_head
        self.model = augmented_model.to(device)
        
        # Loss functions
        self.regression_loss_fn = nn.L1Loss(reduction='mean')
        self.contrastive_loss_fn = NTXentLoss()
        
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
    
    def extract_patch_features(self, patches, ground_truth):
        """Extract features and targets from patches"""
        H_size, W_size = ground_truth.shape
        H_patch, W_patch = H_size // self.patch_size, W_size // self.patch_size
        
        all_embeddings = []
        all_targets = []
        
        for n, patch in enumerate(patches):
            patch = patch.to(self.device).float()
            nb_patches_h, nb_patches_w = patch.shape[-2] // self.patch_size, patch.shape[-1] // self.patch_size
            
            # Extract embeddings
            embeddings = self.model[0].forward_features(patch)["x_norm_patchtokens"]
            embeddings = embeddings.reshape(nb_patches_h, nb_patches_w, self.feat_dim)
            central_embedding = embeddings[nb_patches_h//2, nb_patches_w//2]
            all_embeddings.append(central_embedding)
            
            # Calculate target
            h_coord = n // W_patch
            w_coord = n % W_patch
            gt_patch = ground_truth[h_coord:h_coord + self.patch_size, 
                                  w_coord:w_coord + self.patch_size]
            target = gt_patch.mean()
            all_targets.append(target)
        
        all_embeddings_tensor = torch.stack(all_embeddings)
        all_targets_tensor = torch.stack(all_targets)
        
        all_embeddings_tensor.requires_grad = True
        all_targets_tensor.requires_grad = True
        
        return all_embeddings_tensor, all_targets_tensor
    
    def train_epoch(self, train_set, optimizer, training_index, batch_size=32):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for file, patches, ground_truth in train_set:
            ground_truth = ground_truth.to(self.device).float()
            
            # Extract features
            embeddings, targets = self.extract_patch_features(patches, ground_truth)
            
            if len(embeddings) == 0:
                continue
            
            # Contrastive loss
            optimizer.zero_grad()
            contrastive_loss = self.contrastive_loss_fn(embeddings, targets)
            contrastive_loss.backward()
            optimizer.step()
            
            # Batch processing for regression
            num_patches = len(embeddings)
            for i in range(0, num_patches, batch_size):
                end_idx = min(i + batch_size, num_patches)
                batch_embeddings = embeddings[i:end_idx]
                batch_targets = targets[i:end_idx]
                
                # Forward pass
                predictions = self.model[1](batch_embeddings).squeeze()
                
                # Handle single prediction case
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if batch_targets.dim() == 0:
                    batch_targets = batch_targets.unsqueeze(0)
                
                # Calculate loss                
                loss = self.regression_loss_fn(predictions.cpu(), batch_targets.cpu())
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses) if epoch_losses else 0.0
    
    def evaluate(self, test_set, use_umap=False, max_samples=None):
        """Evaluate model on test set"""
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
                
                # Post-process prediction
                prediction_bool = prediction.astype(bool)
                inverted_prediction = ~prediction_bool
                biggest_mask = self._keep_largest_continuous_mask(inverted_prediction)
                
                # Calculate F1 score
                ground_truth_np = ground_truth.cpu().numpy()
                #score = f1_score(ground_truth_np.flatten(), prediction.flatten(), average='micro')
                score = jaccard_score(ground_truth_np.flatten(), prediction.flatten())
                scores.append(score)
                
                # Display results
                #display_segmentation(prediction, file, score)
                #display_segmentation(ground_truth.cpu().numpy(), file, score)
                
                samples_processed += 1
        
        avg_score = np.mean(scores) if scores else 0.0
        
        # UMAP visualization
        if use_umap and umap_data['embeddings']:
            self._create_umap_visualization(umap_data, avg_score)
        
        return avg_score
    
    def _predict_image(self, patches, ground_truth, umap_data=None):
        """Predict segmentation for a single image"""
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
            output_value = (output > 0.5).float()
            
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
        """Create UMAP visualization"""
        reducer = umap.UMAP(random_state=42)
        embeddings_2d = reducer.fit_transform(umap_data['embeddings'])
        
        labels = np.array(umap_data['labels'])
        binary_labels = (labels > 0.5).astype(int)
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']  # Rest and PSD
        
        for label, color, name in zip([0, 1], colors, ['Rest', 'PSD']):
            mask = binary_labels == label
            if np.sum(mask) > 0:
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=color, s=1, label=name, alpha=0.6)
        
        plt.legend()
        plt.title(f'UMAP Visualization - Test IoU: {test_score:.4f}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.show()
    
    def train_component(self, training_index, epochs, train_set, test_set, 
                       batch_size=32, eval_frequency=10):
        """Train a specific component (adapter or MLP head)"""
        component_names = ['DINOv2 Adapter', 'MLP Head']
        print(f"Training {component_names[training_index]}...")
        
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
                test_score = self.evaluate(test_set, max_samples=max_samples)
                test_scores.append(test_score)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, IoU: {test_score:.4f}")
            else:
                # Reuse last test score for non-evaluation epochs
                test_scores.append(test_scores[-1] if test_scores else 0.0)
        
        return train_losses, test_scores


def save_checkpoint(model, loss_lists, test_loss_lists, iteration, filename=None):
    """Save model checkpoint with training statistics"""
    if filename is None:
        filename = f"checkpoint_iter_{iteration}.pth"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss_lists': loss_lists,
        'test_loss_lists': test_loss_lists,
        'iteration': iteration
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def plot_training_results(loss_lists, test_lists):
    """Plot comprehensive training results"""
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
    
    # Test F1 score plot
    if test_lists[0]:
        x_dino = list(range(1, len(test_lists[0]) + 1))
        ax2.plot(x_dino, test_lists[0], label='DINOv2 Test IoU', 
                color='red', marker='o', linewidth=2)
    
    if test_lists[1]:
        x_mlp = list(range(1, len(test_lists[1]) + 1))
        ax2.plot(x_mlp, test_lists[1], label='MLP Test IoU', 
                color='orange', marker='s', linewidth=2)
    
    ax2.set_ylabel('Test IoU Score')
    ax2.set_xlabel('Iterations')
    ax2.set_title('Test IoU Score Over Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_final_results(loss_lists, test_lists):
    """Print comprehensive final results"""
    print(f"\n{'='*50}")
    print(f"FINAL TRAINING RESULTS")
    print(f"{'='*50}")
    
    component_names = ['DINOv2 Adapter', 'MLP Head']
    
    for i, name in enumerate(component_names):
        if loss_lists[i] and test_lists[i]:
            final_loss = loss_lists[i][-1]
            final_f1 = test_lists[i][-1]
            print(f"{name} - Train Loss: {final_loss:.4f}, Test IoU: {final_f1:.4f}")
            
            # Calculate improvement
            if len(test_lists[i]) > 1:
                improvement = test_lists[i][-1] - test_lists[i][0]
                print(f"{name} IoU Improvement: {improvement:+.4f}")


def run_fine_tuning(nb_iterations=2, nb_epochs_per_iteration=10, 
                   nb_best_patches=100, padding_size=2):
    """Main fine-tuning function"""
    print("Initializing fine-tuner...")
    
    # Initialize fine-tuner
    fine_tuner = FineTuner(augmented_model, device, feat_dim, resize_size)
    
    # Load datasets
    print("Loading datasets...")
    train_set = list(get_data_generator(
        split='training', 
        nb_best_patches=nb_best_patches,
        resize_size=resize_size, 
        padding_size=padding_size, 
        test_proportion=0.2,
        seed=42)
    )
    test_set = list(get_data_generator(
        split='test',
        nb_best_patches=nb_best_patches,
        resize_size=resize_size, 
        padding_size=padding_size, 
        test_proportion=0.2,
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
        print(f"Iteration {iteration} - Training {'MLP Head' if training_index == 1 else 'DINOv2 Adapter'}")
        print(f"{'='*80}")
        
        # Train component
        train_losses, test_scores = fine_tuner.train_component(
            training_index=training_index,
            epochs=nb_epochs_per_iteration,
            train_set=train_set,
            test_set=test_set,
            batch_size=32
        )
        
        # Store results
        final_loss = train_losses[-1] if train_losses else 0.0
        final_score = test_scores[-1] if test_scores else 0.0
        
        loss_lists[training_index].append(final_loss)
        test_lists[training_index].append(final_score)
        
        component_name = 'MLP Head' if training_index == 1 else 'DINOv2 Adapter'
        print(f'{component_name} | Iteration: {iteration} | Loss: {final_loss:.4f} | IoU: {final_score:.4f}')
        
        # Save checkpoint
        if i % 2 == 1:  # After both components trained
            #save_checkpoint(fine_tuner.model, loss_lists, test_lists, iteration)
            iteration += 1
    
    # Final evaluation with UMAP
    print("\nRunning final evaluation with UMAP...")
    final_score = fine_tuner.evaluate(test_set, use_umap=True)
    print(f"Final Test IoU Score: {final_score:.4f}")
    
    # Save final model
    #save_checkpoint(fine_tuner.model, loss_lists, test_lists, "final")
    
    # Plot and print results
    plot_training_results(loss_lists, test_lists)
    print_final_results(loss_lists, test_lists)
    
    return loss_lists, test_lists, fine_tuner


if __name__ == '__main__':
    print("Starting fine-tuning process...")
    
    # Run fine-tuning with optimized parameters
    loss_lists, test_lists, trained_model = run_fine_tuning(
        nb_iterations=1,  # Reduced for testing
        nb_epochs_per_iteration=5,  # Reduced for faster training
        nb_best_patches=10,
        padding_size=2
    )
    
    print("\nFine-tuning completed successfully!")