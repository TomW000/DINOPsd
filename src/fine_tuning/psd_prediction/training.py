import torch 
from torch import nn
from tqdm import tqdm
import numpy as np
import os 
import matplotlib.pyplot as plt

from .head import Psd_Pred_MLP_Head
from .dataset import cross_validation_datasets_generator, nb_best_patches, dataset_bias
from src.fine_tuning.display_results import training_curve, confusion_matrix

from src.setup import device, feat_dim, model_weights_path

# the training/testing loop should train on the cross-validated datasets for one epoch, pick the worst case and train for a few more epochs on it.
# It should then return a trained MLP Head that can later be used for semantic segmentation 

# TODO: Add negative anchors
# TODO: plot similarity histogram
# TODO: how about splitting the dataset wrt euclidian distance to any reference vectors

def detection_head_training(nb_epochs, detection_head, train_set, test_set, return_statistics):

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

    loss_list = []
    prediction_list = []
    test_accuracies = []
    
    optimizer = torch.optim.Adam(detection_head.parameters(), lr=3e-4)
    loss_fn = nn.BCELoss()

    for _ in tqdm(range(nb_epochs), desc='--> Epochs'):
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
            for embeddings, ground_truths in tqdm(test_set, desc='-> Testing', leave=False):
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

        detection_head.train()

    if return_statistics:
        return loss_list, prediction_list, test_accuracies, detection_head.state_dict()
            
    else:
        return test_accuracies[-1]


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
                                                return_statistics=False) # type: ignore
        latest_test_accuracy_list.append(test_accuracy)
        print(f'-Test accuracy: {test_accuracy}')
        if test_accuracy < latest_test_accuracy: # type: ignore
            latest_test_accuracy = test_accuracy
    
    challenging_idx = np.argmin(latest_test_accuracy_list)
    challenging_train_set, challenging_test_set = next((train_set, test_set) for i, (train_set, test_set) in enumerate(dataset_generator) if i==challenging_idx)  
    assert challenging_train_set is not None, "No dataset was selected as the most challenging."
    print(f'-Challenging dataset: {challenging_idx}, worst test accuracy: {min(latest_test_accuracy_list)}')

    plt.figure(figsize=(10,5))
    plt.scatter([i for i in range(len(latest_test_accuracy_list))], latest_test_accuracy_list)
    plt.xlabel('Datasets')
    plt.ylabel('Test accuracy')
    plt.title(f'Test accuracy per dataset - number of PSD patches per image={nb_best_patches} - dataset bias={dataset_bias}')
    ax = plt.gca()
    ax.set_ylim(0, 105)
    plt.show()

    detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim)
    detection_head.to(device)
    nb_epochs = 50
    loss_list, prediction_list, test_accuracies, head_weights = detection_head_training(nb_epochs=nb_epochs, # type: ignore
                                                                        detection_head=detection_head, 
                                                                        train_set=challenging_train_set, 
                                                                        test_set=challenging_test_set, 
                                                                        return_statistics=True)

    training_curve(nb_epochs, loss_list, test_accuracies)
    #confusion_matrix(data_type='psd',
    #                 prediction_list=prediction_list, 
    #                 nb_epochs=nb_epochs,
    #                 split='test')

    torch.save(obj=head_weights, f=os.path.join(model_weights_path, 'psd_head_weights.pt'))