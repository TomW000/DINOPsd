import torch 
from torch import nn
from tqdm import tqdm
import numpy as np
import os 
import umap
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from .mlp_head import Psd_Pred_MLP_Head
from .dataset import dataset_generator, Custom_Dataset, LABELLED_REST, nb_best_patches 
from src.fine_tuning.display_results import training_curve, confusion_matrix

from src.setup import feat_dim, model_weights_path

# the training/testing loop should train on the cross-validated datasets for one epoch, pick the worst case and train for a few more epochs on it.
# It should then return a trained MLP Head that can later be used for semantic segmentation 

# TODO: Add negative anchors
# TODO: plot similarity histogram
# TODO: how about splitting the dataset wrt euclidian distance to any reference vectors


device = 'cpu'


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

                    prediction_list.append([int(predicted_idx), int(true_idx)])

                    if predicted_idx == true_idx:
                        score += 1
                    total += 1
                    
            batch_score = 100 * score / total if total > 0 else 0
            test_accuracies.append(batch_score)


    #UMAP
    if use_umap:
        print('Running UMAP')
        UMAP_EMBEDDINGS = reducer.fit_transform(UMAP_LIST)

        plt.clf()
        unique_labels = np.unique(UMAP_LABELS)
        plt.figure(figsize=(10, 10), dpi=200)
        colors = plt.cm.bwr(np.linspace(0,1,len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = UMAP_LABELS == label
            if label == 0:
                plot_label = 'Rest'
            else:
                plot_label = 'PSD'
            plt.scatter(UMAP_EMBEDDINGS[mask, 0], UMAP_EMBEDDINGS[mask, 1], c=[colors[i]], s=0.5, label=plot_label, marker='o')
        plt.legend()
        plt.title(f'UMAP - Accuracy: {batch_score}')
        ax = plt.gca()
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        plt.show()
        #plt.savefig(os.path.join('/home/tomwelch/Cambridge/Figures', f'UMAP_{batch_score:.2f}.png'))


    if return_statistics:
        return loss_list, prediction_list, test_accuracies, detection_head.state_dict()

    else:
        return test_accuracies[-1]

def filter_dataset(dataset_generator, only_rest: bool = True,
                   test_accuracy_list: list[float] = [], 
                   accuracy_threshold: float = 80.0):
    """
    Filter datasets below an accuracy threshold to focus training on more difficult examples.
    """

    test_accuracy_array = np.array(test_accuracy_list)
    below_threshold_indices = np.where(test_accuracy_array < accuracy_threshold)[0]

    print(f'Number of datasets below threshold: {len(below_threshold_indices)}')

    most_challenging_idx = int(np.argmin(test_accuracy_array))
    best_idx = int(np.argmax(test_accuracy_array))

    rest_train_inputs, rest_train_targets = [], []
    psd_train_inputs = []

    rest_test_inputs, rest_test_targets = [], []
    psd_test_inputs = []

    for i in below_threshold_indices:
        train_loader, test_loader, _ = dataset_generator[i]

        # Collect train data
        for inputs, targets in train_loader:
            inputs, targets = inputs.cpu(), targets.cpu()
            with torch.no_grad():
                mask_rest = targets == 0
                mask_psd = targets == 1

                if mask_rest.any():
                    rest_train_inputs.append(inputs[mask_rest])
                    rest_train_targets.append(targets[mask_rest])
                if not only_rest and mask_psd.any():
                    psd_train_inputs.append(inputs[mask_psd])

        # Collect test data
        for inputs, targets in test_loader:
            inputs, targets = inputs.cpu(), targets.cpu()
            with torch.no_grad():
                mask_rest = targets == 0
                mask_psd = targets == 1

                if mask_rest.any():
                    rest_test_inputs.append(inputs[mask_rest])
                    rest_test_targets.append(targets[mask_rest])
                if not only_rest and mask_psd.any():
                    psd_test_inputs.append(inputs[mask_psd])

    print(f'-Challenging dataset: {most_challenging_idx}, worst test accuracy: {test_accuracy_array[most_challenging_idx]}')
    print(f'-Easiest dataset: {best_idx}, best test accuracy: {test_accuracy_array[best_idx]}')

    if only_rest:
        all_inputs = torch.cat(rest_train_inputs + rest_test_inputs, dim=0)
        all_targets = torch.cat(rest_train_targets + rest_test_targets, dim=0)
        return list(zip(all_inputs, all_targets))
    else:
        all_inputs = torch.cat(psd_train_inputs + psd_test_inputs + rest_train_inputs + rest_test_inputs, dim=0)
        unique_inputs = torch.unique(all_inputs, dim=0)
        psd_targets = torch.ones(unique_inputs.size(0))
        return list(zip(unique_inputs, psd_targets))


    #This returns the datasets below 80% accuracy
def sub_optimal_filter_dataset(dataset_generator, only_rest: bool=True, test_accuracy_list: list[float]=[], accuracy_threshold: float=80.0):

    """
    This function takes a dataset generator and filters out the datasets with a test accuracy below a certain threshold.
    It returns a new dataset generator containing only the filtered datasets.
    The function is used to filter out datasets with a low accuracy in order to focus on the most challenging ones.
    Args:
        dataset_generator (list): A list of datasets to be filtered.
        only_rest (bool): If true, only the rest datasets are filtered. If false, both rest and psd datasets are filtered.
        test_accuracy_list (list): A list of test accuracies for each dataset in the dataset generator.
        accuracy_threshold (float): The minimum test accuracy required for a dataset to be included in the filtered dataset generator.
    Returns:
        list: A new dataset generator containing only the filtered datasets.
    """

    accuracy_threshold = 80.0
    latest_test_accuracies = np.array(test_accuracy_list)
    masks = (latest_test_accuracies < accuracy_threshold)
    below_threshold_indices = np.where(masks)[0]
    print(f'Number of datasets below threshold: {len(below_threshold_indices)}')

    most_challenging_idx = np.argmin(test_accuracy_list)
    best_idx = np.argmax(test_accuracy_list)

    rest_train_inputs, rest_train_targets = [], []
    psd_train_inputs = []

    rest_test_inputs, rest_test_targets = [], []
    psd_test_inputs = []

    for i, (train_loader, test_loader, _) in enumerate(dataset_generator):
        if i in below_threshold_indices:
            for batch in train_loader:
                inputs, targets = batch
                for input, target in zip(inputs, targets):
                    if target.item() == 0.0:
                        rest_train_inputs.append(input)
                        rest_train_targets.append(target)
                    elif target.item() == 1.0:
                        psd_train_inputs.append(input)
                    else:
                        print(f'Unexpected target value: {target}')
                        
            for batch in test_loader:
                inputs, targets = batch
                for input, target in zip(inputs, targets):
                    if target.item() == 0.0:
                        rest_test_inputs.append(input)
                        rest_test_targets.append(target)
                    elif target.item() == 1.0:
                        psd_test_inputs.append(input)
                    else:
                        print(f'Unexpected target value: {target}')

    if type(dataset_generator) == list:
        dataset_generator.clear() 

    print(f'-Challenging dataset: {most_challenging_idx}, worst test accuracy: {min(test_accuracy_list)}')
    print(f'-Easiest dataset: {best_idx}, best test accuracy: {max(test_accuracy_list)}')

    if only_rest:
        inputs = torch.from_numpy(np.concatenate([rest_train_inputs, rest_test_inputs]))
        targets = torch.from_numpy(np.concatenate([rest_train_targets, rest_test_targets]))
        
        # this dataset is solely used to perform the second filtering step - otherwise, we could have directly used the original Custom_Dataset
        assert inputs is not None and targets is not None, "No filtered dataset was selected."
        
        FILTERED_LABELLED_REST_SET = list(zip(inputs, targets))
        return FILTERED_LABELLED_REST_SET

    else:
        # below threshold after filtering means actually psd
        psd_inputs = torch.from_numpy(np.concatenate([psd_train_inputs, psd_test_inputs, rest_train_inputs, rest_test_inputs]))
        unique_inputs = torch.unique(psd_inputs, dim=0)
        psd_targets = torch.ones(unique_inputs.shape[0])
        
        assert psd_inputs is not None and unique_inputs is not None and psd_targets is not None, "No filtered dataset was selected."
        
        FILTERED_LABELLED_PSD_SET = list(zip(unique_inputs, psd_targets))        
        return FILTERED_LABELLED_PSD_SET

def training_main():
    """
    Train a psd detection head on a cross-validation dataset until the test accuracy 
    does not improve any more. The most challenging dataset is the one with the lowest 
    test accuracy after training for one epoch. The MLP head is then trained on this 
    dataset for 10 epochs and the training curve is plotted. The test accuracy, the 
    confusion matrix and the trained head are saved.
    """
    dataset_g = list(dataset_generator(sliding_window=False,
                                       test_proportion=0.2,
                                       train_batch_size=50, 
                                       test_batch_size=50,
                                       seed=42))

    latest_test_accuracy = 100.0
    latest_test_accuracy_list = []
    for train_set, test_set, _ in tqdm(dataset_g, desc='Looping through datasets', leave=False):
        detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim, nb_outputs=1, use_sigmoid=True)
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

    plt.figure(figsize=(14,10), dpi=200)
    plt.scatter([i for i in range(len(latest_test_accuracy_list))], latest_test_accuracy_list, c='darkblue')
    plt.xlabel('Datasets splits')
    plt.ylabel('Test accuracy')
    plt.title('Balanced test accuracies on entire dataset')
    plt.axhline(y=80, color='r', linestyle='--', label='Threshold = 80%')
    ax = plt.gca()
    ax.set_ylim(0, 105) 
    plt.legend()
    plt.show()
    #plt.savefig(os.path.join('/home/tomwelch/Cambridge/Figures', 'test_accuracy_per_dataset.png')) #TODO: CHANGE BACK

    FILTERED_LABELLED_REST_SET = filter_dataset(dataset_generator=dataset_g,
                                                only_rest=True,                                                
                                                test_accuracy_list=latest_test_accuracy_list, 
                                                accuracy_threshold=80.0)

    filtered_dataset_g = list(dataset_generator(sliding_window=True,
                                           LABELLED_REST_SET=FILTERED_LABELLED_REST_SET,
                                           test_proportion=0.2,
                                           stride=100,
                                           train_batch_size=50, 
                                           test_batch_size=50, 
                                           seed=42))

    latest_test_accuracy_list = []  
    for train_set, test_set, _ in tqdm(filtered_dataset_g, desc='Looping through filtered datasets', leave=False):
        detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim, nb_outputs=1, use_sigmoid=True)
        detection_head.to(device)
        test_accuracy = detection_head_training(nb_epochs=1, 
                                                detection_head=detection_head,
                                                train_set=train_set, 
                                                test_set=test_set, 
                                                return_statistics=False,
                                                use_umap=False)
        latest_test_accuracy_list.append(test_accuracy)
        print(f'-Test accuracy: {test_accuracy}')

    plt.figure(figsize=(10,10), dpi=200)
    plt.scatter([i for i in range(len(latest_test_accuracy_list))], latest_test_accuracy_list, c='darkgreen')
    plt.xlabel('Datasets splits')
    plt.ylabel('Test accuracy')
    plt.title('Balanced test accuracies on filtred dataset')
    plt.axhline(y=70, color='r', linestyle='--', label='New threshold = 70%')
    ax = plt.gca()
    ax.set_ylim(0, 105) 
    plt.legend()
    plt.show()
    #plt.savefig(os.path.join('/home/tomwelch/Cambridge/Figures', 'test_accuracy_per_filtered_dataset.png')) #TODO: CHANGE BACK
    
    AUGMENTED_LABELLED_PSD_SET = filter_dataset(dataset_generator=filtered_dataset_g,
                                          only_rest=False, 
                                          test_accuracy_list=latest_test_accuracy_list, 
                                          accuracy_threshold=70.0)
    
    EASIEST_LABELLED_REST_SET = LABELLED_REST[-len(AUGMENTED_LABELLED_PSD_SET):] 

    print(f'{len(AUGMENTED_LABELLED_PSD_SET)} PSD patches')
    print(f'{len(EASIEST_LABELLED_REST_SET)} rest patches')
    
    augmemted_training_dataset = Custom_Dataset(
        LABELLED_PSD=AUGMENTED_LABELLED_PSD_SET,
        LABELLED_REST=EASIEST_LABELLED_REST_SET,
        set_type='training',
        test_proportion=0.2,
        )
                
    augmemted_test_dataset = Custom_Dataset(
        LABELLED_PSD=AUGMENTED_LABELLED_PSD_SET,
        LABELLED_REST=EASIEST_LABELLED_REST_SET,
        set_type='test',
        test_proportion=0.2,
        )
    
    train_loader = torch.utils.data.DataLoader(augmemted_training_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(augmemted_test_dataset, batch_size=100, shuffle=True)
    
    
    detection_head = Psd_Pred_MLP_Head(device=device, feat_dim=feat_dim, nb_outputs=1, use_sigmoid=True)
    detection_head.to(device)
    nb_epochs = 50
    loss_list, prediction_list, test_accuracies, head_weights = detection_head_training(nb_epochs=nb_epochs, # type: ignore
                                                                        detection_head=detection_head, 
                                                                        train_set=train_loader, 
                                                                        test_set=test_loader,
                                                                        return_statistics=True,
                                                                        use_umap=True)

    training_curve(nb_epochs, loss_list, test_accuracies)
    confusion_matrix(data_type='psd',
                     prediction_list=prediction_list, 
                     nb_epochs=nb_epochs,
                     split='test')

    torch.save(obj=head_weights, f=os.path.join(model_weights_path, 'psd_head_weights.pt')) #TODO: CHANGE BACK
    
    #return train_loader, test_loader


#train_loader, test_loader = training_main()

if __name__ == '__main__':
    training_main()
