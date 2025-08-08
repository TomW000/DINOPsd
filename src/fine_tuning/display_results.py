import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.setup import neurotransmitters, resize_size
from src.perso_utils import load_image
from src.analysis_utils import resize_hdf_image


def training_curve(nb_epochs, loss_list, test_accuracies):
    """Plot training loss and test accuracy over epochs."""
    x = list(range(nb_epochs))
    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=150)
    ax2 = ax1.twinx()

    # Plot curves
    lns1 = ax1.plot(x, loss_list, label='Train Loss', color='blue')
    ax1.set_ylim(0, max(loss_list) * 1.05)
    ax1.set_ylabel('Train Loss')

    lns2 = ax2.plot(x, test_accuracies, label='Test Accuracy', color='red')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Test Accuracy')

    # Combine legends
    lns = lns1 + lns2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc=0)

    ax1.set_xlabel('Epochs')
    plt.title('Training Curve')
    plt.tight_layout()
    plt.show()


def class_proportions(data_type, proportion_list):
    """Visualize class proportions as a heatmap with counts and percentages."""
    nb_classes = len(proportion_list[0])
    one_hot = np.eye(nb_classes)

    # Flatten and count class occurrences
    gts = np.concatenate(proportion_list)
    vectors, counts = np.unique(gts, axis=0, return_counts=True)
    positions = [np.where(np.all(one_hot == v, axis=1)) for v in vectors]

    proportions = np.zeros((nb_classes, 1))
    for count, pos in zip(counts, positions):
        proportions[pos] = count

    proportions = 100 * proportions / np.sum(proportions)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    img = ax.imshow(proportions.T, cmap='RdYlGn')

    for k, prop in enumerate(proportions):
        ax.text(x=k, y=0,
                s=f'{round(prop.item(), 2)}%\n({int(proportions[k] * np.sum(counts) / 100)})',
                ha="center", va="center", color="black")

    if data_type == 'neuro':
        ax.set_xticks(range(nb_classes), labels=neurotransmitters, rotation=-45, ha="right", rotation_mode="anchor")
    elif data_type == 'psd':
        ax.set_xticks(range(nb_classes), labels=['Psd', 'Rest'], rotation=-45, ha="right", rotation_mode="anchor")

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.text(nb_classes, 0, f'Total: {int(np.sum(counts))}', va='center', ha='left', color='black')

    ax.set_title('Class Proportions')
    ax.set_yticks([])

    fig.colorbar(img, ax=ax, orientation='horizontal', label='Proportion (%)')
    plt.tight_layout()
    plt.show()


def confusion_matrix(data_type, prediction_list, nb_epochs, split):
    """Plot confusion matrix with percentages and raw counts."""
    
    if data_type == 'neuro':
        labels = neurotransmitters
        nb_classes = len(labels)
    elif data_type == 'psd':
        labels = ['Psd', 'Rest']
        nb_classes = len(labels)

    conf_matrix = np.zeros((nb_classes, nb_classes), dtype=int)

    for pred, truth in prediction_list:
        conf_matrix[truth, pred] += 1

    total_matrix = conf_matrix.copy()
    norm_matrix = np.zeros_like(conf_matrix, dtype=float)

    for i, row in enumerate(conf_matrix):
        row_sum = np.sum(row)   
        if row_sum > 0:
            norm_matrix[i] = 100 * row / row_sum

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    im = ax.imshow(norm_matrix, cmap='YlGn')

    ax.set_xticks(range(nb_classes), labels=labels, rotation=-45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(nb_classes), labels=labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(nb_classes):
        for j in range(nb_classes):
            value = norm_matrix[i, j]
            count = total_matrix[i, j]
            ax.text(j, i, f'{value:.2f}%\n({count})', ha='center', va='center', color='black')

    # Row and column totals
    for i in range(nb_classes):
        ax.text(nb_classes, i, f'({np.sum(total_matrix[i])})', va='center', ha='left', color='black')
    for j in range(nb_classes):
        ax.text(j, nb_classes, f'({np.sum(total_matrix[:, j])})', va='center', ha='center', color='black')

    ax.text(nb_classes, nb_classes, f'({np.sum(total_matrix)})', va='center', ha='left', color='black')

    ax.set_title(f'Confusion Matrix - {split} - Epochs={nb_epochs}')
    plt.tight_layout()
    plt.show()


def display_segmentation(mask, file, score):

    mask = np.array(mask)
    
    h, w = mask.shape
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    initial_image = load_image(file)[0] if type(file) == str else file
    resized_image = resize_hdf_image(initial_image, resize_size=resize_size)

    ax.imshow(resized_image, cmap='gray', extent=[0, h, w, 0])

    # Overlay the heatmap
    sns.heatmap(
        mask,
        cmap='bwr',
        alpha=0.5,             # Make the heatmap semi-transparent
        ax=ax,
        cbar=False,
        xticklabels=False,
        yticklabels=False
    )

    plt.title("Segmentation - Score={:.2f}".format(score))
    plt.tight_layout()
    plt.show()