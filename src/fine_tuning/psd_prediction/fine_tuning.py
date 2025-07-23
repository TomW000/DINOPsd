import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import f1_score

from src.setup import model, device, model_weights_path, resize_size, feat_dim
from src.fine_tuning.adaptor.AdaptFormer import augmented_model
from src.fine_tuning.display_results import display_segmentation
from .sliding_window_dataset import get_data_generator

def training_block(model,
                   index, 
                   nb_epochs, 
                   train_set, 
                   test_set, 
                   return_statistics=False, 
                   use_umap=False,
                   batch_size=32):  # Added batch processing parameter

    # Set up trainable parameters and optimizer
    if index == 1:
        # Train MLP head
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model[1].parameters():
            param.requires_grad = True 

        optimizer = torch.optim.Adam(model[1].parameters(), lr=3e-4)
        print("Training MLP Head")
        
    elif index == 0:
        # Train DINOv2 adapter layers
        trainable_params = []
        
        for param in model.parameters():
            param.requires_grad = False 
        
        for k in range(len(list(model[0].blocks))):
            block = model[0].blocks[k].mlp
        
            for param in block.down_proj.parameters(): 
                param.requires_grad = True 
                trainable_params.append(param)
            
            for param in block.up_proj.parameters():
                param.requires_grad = True 
                trainable_params.append(param)
        
        print(f"Training DINOv2 adapter layers - {len(trainable_params)} parameters")
        optimizer = torch.optim.Adam(trainable_params, lr=3e-4)
        
    model.to(device)

    # UMAP setup
    if use_umap:
        reducer = umap.UMAP(random_state=42)

    loss_list = []
    test_score_list = []

    patch_size = model[0].patch_size
    
    # Use BCEWithLogitsLoss for binary classification
    loss_fn = nn.BCEWithLogitsLoss()

    def accuracy_fn(y_true, y_pred): 
        # Convert tensors to numpy for sklearn
        y_true_np = y_true.cpu().detach().numpy().flatten()
        y_pred_np = y_pred.cpu().detach().numpy().flatten()
        # Binarize predictions
        y_pred_binary = (y_pred_np > 0.5).astype(int)
        y_true_binary = (y_true_np > 0.5).astype(int)
        return f1_score(y_true_binary, y_pred_binary, zero_division=0)

    for epoch in tqdm(range(nb_epochs), desc=f'--> Epochs (Index {index})'):
        model.train()
        epoch_loss_list = []

        for file, patches, ground_truth in train_set:
            ground_truth = ground_truth.to(device).float()
            H_size, W_size = ground_truth.shape
            H_patch, W_patch = H_size // patch_size, W_size // patch_size
            
            # OPTIMIZED: Collect all embeddings and targets first
            all_embeddings = []
            all_targets = []
            
            # Extract embeddings for all patches (more efficient batching)
            for n, patch in enumerate(patches):
                patch = patch.to(device).float()
                nb_patches_h, nb_patches_w = patch.shape[-2] // patch_size, patch.shape[-1] // patch_size
                
                # Get embeddings - use no_grad for feature extraction to save memory
                with torch.no_grad() if index == 1 else torch.enable_grad():
                    embeddings = model[0].forward_features(patch)["x_norm_patchtokens"].reshape(nb_patches_h, nb_patches_w, feat_dim)
                    central_embedding = embeddings[nb_patches_h//2, nb_patches_w//2]
                    all_embeddings.append(central_embedding)
                
                # Calculate coordinates and target
                h_coord = n // W_patch
                w_coord = n % W_patch
                gt_patch = ground_truth[h_coord:h_coord + patch_size, w_coord:w_coord + patch_size]
                target = gt_patch.mean()
                all_targets.append(target)
            
            # OPTIMIZED: Batch processing
            if all_embeddings:
                embeddings_tensor = torch.stack(all_embeddings)
                targets_tensor = torch.stack(all_targets)
                
                # Process in smaller batches to manage memory
                num_patches = len(embeddings_tensor)
                for i in range(0, num_patches, batch_size):
                    end_idx = min(i + batch_size, num_patches)
                    batch_embeddings = embeddings_tensor[i:end_idx]
                    batch_targets = targets_tensor[i:end_idx]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass through MLP
                    predictions = model[1](batch_embeddings).squeeze()
                    
                    # Handle single prediction case
                    if predictions.dim() == 0:
                        predictions = predictions.unsqueeze(0)
                    if batch_targets.dim() == 0:
                        batch_targets = batch_targets.unsqueeze(0)
                    
                    # Calculate loss
                    loss = loss_fn(predictions.cpu(), batch_targets.cpu())
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss_list.append(loss.item())
        if epoch_loss_list:  # Only append if we have losses
            loss_list.append(np.mean(epoch_loss_list))
        else:
            loss_list.append(0.0)

        # Evaluation phase - OPTIMIZED: Less frequent evaluation for speed
        if epoch % max(1, nb_epochs // 10) == 0 or epoch == nb_epochs - 1:
            model.eval()
            with torch.no_grad():
                # UMAP setup for first epoch only
                if use_umap and epoch == 0:
                    UMAP_LIST, UMAP_PREDICTED_LABELS = [], []

                epoch_scores = []
                
                # OPTIMIZED: Limit number of test samples during training for speed
                test_samples_processed = 0
                max_test_samples = 50 if epoch < nb_epochs - 1 else float('inf')  # Full evaluation only on last epoch
                for file, patches, ground_truth in test_set:
                    if test_samples_processed >= max_test_samples:
                        break
                    
                    ground_truth = ground_truth.to(device).float()
                    H_size, W_size = ground_truth.shape
                    H_patch, W_patch = H_size // patch_size, W_size // patch_size
                    prediction = torch.zeros(ground_truth.shape, device=device)
                    
                    for n, patch in enumerate(patches):
                        patch = patch.to(device).float()
                        nb_patches_h, nb_patches_w = patch.shape[-2] // patch_size, patch.shape[-1] // patch_size
                        embeddings = model[0].forward_features(patch)["x_norm_patchtokens"].reshape(nb_patches_h, nb_patches_w, feat_dim)
                        central_embedding = embeddings[nb_patches_h//2, nb_patches_w//2]
                        central_embedding = central_embedding.unsqueeze(0)
                        
                        output = model[1](central_embedding)
                        output_sigmoid = torch.sigmoid(output.squeeze())
                        output_value = (output > 0.5).float()
                        # UMAP data collection
                        if use_umap and epoch == 0:    
                            UMAP_LIST.append(central_embedding.cpu().numpy().flatten())
                            UMAP_PREDICTED_LABELS.append(output_value.cpu().numpy())
                        
                        h_coord = (n // W_patch) * patch_size
                        w_coord = (n % W_patch) * patch_size
                        prediction[h_coord:h_coord + patch_size, w_coord:w_coord + patch_size] = output_value

                    display_segmentation(ground_truth.cpu().numpy(), file)

                    # Calculate F1 score for this sample
                    test_score = accuracy_fn(ground_truth, prediction)
                    epoch_scores.append(test_score)
                    test_samples_processed += 1
                    
                if epoch_scores:
                    test_score_list.append(np.mean(epoch_scores))
                else:
                    test_score_list.append(0.0)
        else:
            # Skip evaluation for intermediate epochs, reuse last score
            if test_score_list:
                test_score_list.append(test_score_list[-1])
            else:
                test_score_list.append(0.0)
                
        # Print epoch progress - less frequent printing
        if epoch % max(1, nb_epochs // 5) == 0 or epoch == nb_epochs - 1:
            print(f"Epoch {epoch+1}/{nb_epochs} - Loss: {loss_list[-1]:.4f}, F1: {test_score_list[-1]:.4f}")

    # UMAP visualization - only if requested and data available
    if use_umap and 'UMAP_LIST' in locals() and len(UMAP_LIST) > 0:
        print('Running UMAP')
        UMAP_EMBEDDINGS = reducer.fit_transform(UMAP_LIST)

        plt.figure(figsize=(10, 8))
        
        # Convert to numpy arrays for easier handling
        UMAP_PREDICTED_LABELS = np.array(UMAP_PREDICTED_LABELS)
        
        # Create binary labels for coloring
        binary_labels = (UMAP_PREDICTED_LABELS > 0.5).astype(int)
        
        unique_labels = np.unique(binary_labels)
        colors = ['blue', 'red']  # Rest and PSD
        
        for i, label in enumerate(unique_labels):
            mask = binary_labels == label
            plot_label = 'Rest' if label == 0 else 'PSD'
            if np.sum(mask) > 0:  # Only plot if there are points
                plt.scatter(UMAP_EMBEDDINGS[mask, 0], UMAP_EMBEDDINGS[mask, 1], 
                           c=colors[i], s=0.5, label=plot_label, marker='o', alpha=0.6)
        
        plt.legend()
        plt.title(f'UMAP - Final Test Score: {test_score_list[-1]:.4f}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.show()

    if return_statistics:
        return loss_list, [], test_score_list, model.state_dict()  # Empty prediction_list
    else:
        return loss_list[-1], test_score_list[-1]


def fine_tuning(model, 
                nb_iterations, 
                nb_epochs_per_iteration,
                padding_size):

    block_names = ['DINOv2', 'MLP Head']
    index_list = []
    for _ in range(nb_iterations):
        index_list.extend([1, 1])  # Train MLP head first, then DINOv2
        
    general_loss_list = [[], []]  # [DINOv2, MLP Head]
    general_test_loss_list = [[], []]

    # Get datasets with proper train/validation split
    print("Loading training data...")
    train_set = get_data_generator(split='training', 
                                  nb_best_patches=10,
                                  resize_size=resize_size, 
                                  padding_size=padding_size, 
                                  test_proportion=0,  # Use all training data
                                  seed=42)

    print("Loading test data...")
    test_set = get_data_generator(split='test',  # Use validation split for testing
                                 nb_best_patches=10,
                                 resize_size=resize_size, 
                                 padding_size=padding_size, 
                                 test_proportion=1,   # Use all validation data
                                 seed=42)

    iteration = 1
    for i, index in enumerate(tqdm(index_list, desc='Training Iterations')):
        print(f"\n==================================== Training {block_names[index]} - Iteration {iteration} ====================================")

        # CRITICAL FIX: Use the actual index variable, not hardcoded 1
        loss, test_loss = training_block(model=model,
                                       index=index,
                                       nb_epochs=nb_epochs_per_iteration, 
                                       train_set=train_set, 
                                       test_set=test_set, 
                                       return_statistics=False, 
                                       use_umap=(i == 0))  # Only use UMAP for first iteration

        print(f'Block trained: {block_names[index]} | Iteration: {iteration} | Train loss: {loss:.4f} | Test F1: {test_loss:.4f}')
        
        general_loss_list[index].append(loss)
        general_test_loss_list[index].append(test_loss)
        
        # Only increment iteration counter after both blocks are trained
        if index == 0:  # DINOv2 is trained second in each iteration
            iteration += 1
        
    return general_loss_list, general_test_loss_list 


def save_model_checkpoint(model, loss_lists, test_loss_lists, iteration):
    """Save model checkpoint with training statistics"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss_lists': loss_lists,
        'test_loss_lists': test_loss_lists,
        'iteration': iteration
    }
    
    checkpoint_path = f"checkpoint_iter_{iteration}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == '__main__':
    print("Starting fine-tuning process...")
    
    # OPTIMIZED: Reduced iterations and epochs for faster training
    general_loss_list, general_test_loss_list = fine_tuning(model=augmented_model,
                                                           nb_iterations=1,  # Reduced for testing
                                                           nb_epochs_per_iteration=2,
                                                           padding_size=1)  # Reduced for faster training

    # Save final model
    save_model_checkpoint(augmented_model, general_loss_list, general_test_loss_list, "final")

    # Enhanced plotting
    x_dino = [i+1 for i in range(len(general_loss_list[0]))]
    x_mlp = [i+1 for i in range(len(general_loss_list[1]))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)

    # Training loss plot
    if general_loss_list[0]:  # Check if DINOv2 was trained
        ax1.plot(x_dino, general_loss_list[0], label='DINOv2 Train Loss', 
                color='blue', marker='o', linewidth=2)
    if general_loss_list[1]:  # Check if MLP was trained
        ax1.plot(x_mlp, general_loss_list[1], label='MLP Train Loss', 
                color='cyan', marker='s', linewidth=2)
    ax1.set_ylabel('Train Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_title('Training Loss Over Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test F1 score plot
    if general_test_loss_list[0]:
        ax2.plot(x_dino, general_test_loss_list[0], label='DINOv2 Test F1', 
                color='red', marker='o', linewidth=2)
    if general_test_loss_list[1]:
        ax2.plot(x_mlp, general_test_loss_list[1], label='MLP Test F1', 
                color='orange', marker='s', linewidth=2)
    ax2.set_ylabel('Test F1 Score')
    ax2.set_xlabel('Iterations')
    ax2.set_title('Test F1 Score Over Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Print comprehensive final results
    print(f"\n" + "="*50)
    print(f"FINAL TRAINING RESULTS")
    print(f"="*50)
    
    if general_loss_list[0] and general_test_loss_list[0]:
        print(f"DINOv2 - Train Loss: {general_loss_list[0][-1]:.4f}, Test F1: {general_test_loss_list[0][-1]:.4f}")
    if general_loss_list[1] and general_test_loss_list[1]:
        print(f"MLP Head - Train Loss: {general_loss_list[1][-1]:.4f}, Test F1: {general_test_loss_list[1][-1]:.4f}")
    
    # Calculate improvement
    if len(general_test_loss_list[1]) > 1:
        mlp_improvement = general_test_loss_list[1][-1] - general_test_loss_list[1][0]
        print(f"MLP F1 Improvement: {mlp_improvement:+.4f}")
    
    if len(general_test_loss_list[0]) > 1:
        dino_improvement = general_test_loss_list[0][-1] - general_test_loss_list[0][0]
        print(f"DINOv2 F1 Improvement: {dino_improvement:+.4f}")