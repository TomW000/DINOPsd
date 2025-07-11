import torch 
from torch import nn
from tqdm import tqdm
import numpy as np
import os 
import umap
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from .head import Psd_Pred_MLP_Head
from .dataset import cross_validation_datasets_generator, nb_best_patches, dataset_bias
from src.fine_tuning.display_results import training_curve, confusion_matrix

from src.setup import device, feat_dim, model_weights_path

# the training/testing loop should train on the cross-validated datasets for one epoch, pick the worst case and train for a few more epochs on it.
# It should then return a trained MLP Head that can later be used for semantic segmentation 

# TODO: Add negative anchors
# TODO: plot similarity histogram
# TODO: how about splitting the dataset wrt euclidian distance to any reference vectors


def detection_head_training(nb_epochs, detection_head, train_set, test_set, return_statistics, use_umap):

    """
    Train a detection head using the provided training and testing datasets.

    Args:
        nb_epochs (int): Number of epochs to train the model for.
        detection_head (nn.Module): The model to be trained.
        train_set (DataLoader): DataLoader for the training dataset, providing batches of embeddings and one-hot labels.
        test_set (DataLoader): DataLoader for the testing dataset, providing batches of embeddings and one-hot labels.
        return_statistics (bool): If True, returns training loss, prediction list, test accuracies, and model state dict.

    Returns:
        If return_statistics is True:
            tuple: (loss_list, prediction_list, test_accuracies, detection_head.state_dict())
        Else:
            float: The test accuracy of the final epoch.
    """

    #UMAP
    if use_umap:
        reducer = umap.UMAP(random_state=42)

    loss_list = []
    prediction_list = []
    test_accuracies = []
    
    optimizer = torch.optim.Adam(detection_head.parameters(), lr=3e-4)
    loss_fn = nn.BCELoss()

    for epoch in tqdm(range(nb_epochs), desc='--> Epochs'):
        detection_head.train()
        epoch_loss_list = []

        for embeddings, ground_truths in tqdm(train_set, desc='-> Training', leave=False):
            embeddings = embeddings.to(device)
            ground_truths = ground_truths.to(device).float().unsqueeze(1)  # Make sure targets are float for BCELoss

            optimizer.zero_grad()
            outputs = detection_head(embeddings).float() # Ensure float32

            loss = loss_fn(outputs, ground_truths)
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss.item())

        loss_list.append(np.mean(epoch_loss_list))

        # Evaluation phase
        detection_head.eval()
        with torch.no_grad():
            score = 0
            total = 0
            
            #UMAP
            if use_umap and epoch == 0:
                UMAP_LIST, UMAP_LABELS = [], []
                UMAP_EMBEDDINGS = []
                
            for embeddings, ground_truths in tqdm(test_set, desc='-> Testing', leave=False):
                
                if use_umap and epoch == 0:    
                    #UMAP:
                    UMAP_LIST.extend(embeddings)
                    UMAP_LABELS.extend(ground_truths)
                    
                
                embeddings = embeddings.to(device)
                ground_truths = ground_truths.to(device)
                

                outputs = detection_head(embeddings)

                for output, gt in zip(outputs, ground_truths):
                    predicted_idx = torch.round(output)
                    true_idx = gt
                    assert type(predicted_idx) == type(true_idx), "predicted_idx and true_idx must be of the same type"

                    prediction_list.append([predicted_idx, true_idx])

                    if predicted_idx == true_idx:
                        score += 1
                    total += 1
                    
            batch_score = 100 * score / total if total > 0 else 0
            test_accuracies.append(batch_score)


    #UMAP
    if use_umap:
        UMAP_EMBEDDINGS = reducer.fit_transform(UMAP_LIST)

        plt.clf()
        unique_labels = np.unique(UMAP_LABELS)
        
        colors = plt.cm.bwr(np.linspace(0,1,len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = UMAP_LABELS == label
            if label == 0:
                plot_label = 'Rest'
            else:
                plot_label = 'PSD'
            plt.scatter(UMAP_EMBEDDINGS[mask, 0], UMAP_EMBEDDINGS[mask, 1], c=[colors[i]], s=0.5, label=plot_label, marker='o')
        plt.legend()
        plt.title(f'UMAP - accuracy: {batch_score}')
        plt.savefig(os.path.join('/home/tomwelch/Cambridge/Figures', f'UMAP_{batch_score:.2f}.png'))


    if return_statistics:
        return loss_list, prediction_list, test_accuracies, detection_head.state_dict()

    else:
        return test_accuracies[-1]


def rolling_window_dataset_generator(train_inputs, train_targets, test_inputs, test_targets, window_width):
    
    for 
    
    yield DataLoader(filtered_train_dataset, batch_size=50, shuffle=True), DataLoader(filtered_test_dataset, batch_size=50, shuffle=True)




def training_main():
    """
    Train a psd detection head on a cross-validation dataset until the test accuracy 
    does not improve any more. The most challenging dataset is the one with the lowest 
    test accuracy after training for one epoch. The MLP head is then trained on this 
    dataset for 10 epochs and the training curve is plotted. The test accuracy, the 
    confusion matrix and the trained head are saved.
    """
    dataset_generator = list(cross_validation_datasets_generator(test_proportion=0.2))

    latest_test_accuracy = 100.0
    latest_test_accuracy_list = []
    for train_set, test_set in tqdm(dataset_generator, desc='Looping through datasets', leave=False):
        detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
        detection_head.to(device)
        test_accuracy = detection_head_training(nb_epochs=1, 
                                                detection_head=detection_head,
                                                train_set=train_set, 
                                                test_set=test_set, 
                                                return_statistics=False,
                                                use_umap=False)
        latest_test_accuracy_list.append(test_accuracy)
        print(f'-Test accuracy: {test_accuracy}')
        if test_accuracy < latest_test_accuracy: # type: ignore
            latest_test_accuracy = test_accuracy


    plt.figure(figsize=(10,5))
    plt.scatter([i for i in range(len(latest_test_accuracy_list))], latest_test_accuracy_list)
    plt.xlabel('Datasets')
    plt.ylabel('Test accuracy')
    plt.title(f'Test accuracies on entire dataset - number of PSD patches per image={nb_best_patches} - dataset bias={dataset_bias}')
    plt.axhline(y=80, color='r', linestyle='--', label='Threshold')
    ax = plt.gca()
    ax.set_ylim(0, 105) 
    #plt.savefig(os.path.join('/home/tomwelch/Cambridge/Figures', 'test_accuraciy_per_dataset.png')) #TODO: CHANGE BACK


    #This returns the datasets below 80% accuracy

    accuracy_threshold = 80.0
    latest_test_accuracies = np.array(latest_test_accuracy_list)
    masks = (latest_test_accuracies < accuracy_threshold)
    below_threshold_indices = np.argsort(latest_test_accuracies[masks])

    most_challenging_idx = np.argmin(latest_test_accuracy_list)
    best_idx = np.argmax(latest_test_accuracy_list)

    train_inputs = []
    train_targets = []

    test_inputs = []
    test_targets = []

    for i, (train_loader, test_loader) in enumerate(dataset_generator):
        if i in below_threshold_indices:
            for batch in train_loader:
                inputs, targets = batch
                train_inputs.append(inputs)
                train_targets.append(targets)

            for batch in test_loader:
                inputs, targets = batch
                test_inputs.append(inputs)
                test_targets.append(targets)

    # Concatenate all batches
    train_inputs = torch.cat(train_inputs)
    train_targets = torch.cat(train_targets)

    test_inputs = torch.cat(test_inputs)
    test_targets = torch.cat(test_targets)

    # Create datasets and loaders
    diversified_train_dataset = TensorDataset(train_inputs, train_targets)
    diversified_test_dataset = TensorDataset(test_inputs, test_targets)

    diversified_train_set = DataLoader(diversified_train_dataset, batch_size=50, shuffle=True)
    diversified_test_set = DataLoader(diversified_test_dataset, batch_size=50, shuffle=True)

    
    
    assert diversified_train_set and diversified_test_set is not None, "No dataset was selected."
    print(f'-Challenging dataset: {most_challenging_idx}, worst test accuracy: {min(latest_test_accuracy_list)}')
    print(f'-Easiest dataset: {best_idx}, worst test accuracy: {max(latest_test_accuracy_list)}')

    
    detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
    detection_head.to(device)
    nb_epochs = 50
    loss_list, prediction_list, test_accuracies, head_weights = detection_head_training(nb_epochs=nb_epochs, # type: ignore
                                                                        detection_head=detection_head, 
                                                                        train_set=diversified_train_set, 
                                                                        test_set=diversified_test_set,
                                                                        return_statistics=True,
                                                                        use_umap=True)

    training_curve(nb_epochs, loss_list, test_accuracies)
    #confusion_matrix(data_type='psd',
     #                prediction_list=prediction_list, 
      #               nb_epochs=nb_epochs,
       #              split='test')

    torch.save(obj=head_weights, f=os.path.join(model_weights_path, 'psd_head_weights.pt')) #TODO: CHANGE BACK

if __name__ == '__main__':
    training_main()