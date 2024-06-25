"""
Visualization
"""

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_loss_accuracy(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, save_path=None):
    """
    Plot Loss & Accuracy

    :param train_loss_list: train loss list
    :param valid_loss_list: valid loss list
    :param train_accuracy_list: train accuracy list
    :param valid_accuracy_list: valid accuracy list
    :param save_path: save path (default: None for display only)
    """
    plt.figure()
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, len(valid_loss_list) + 1), valid_loss_list, label='Valid Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.show()

    plt.figure()
    plt.plot(range(1, len(train_accuracy_list) + 1), train_accuracy_list, label='Train Accuracy')
    plt.plot(range(1, len(valid_accuracy_list) + 1), valid_accuracy_list, label='Valid Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'accuracy_curve.png'))
    plt.show()


def plot_confusion_matrix(conf_mat, class_names, normalize=True, save_path=None):
    """
    Plot Confusion Matrix

    :param conf_mat: confusion matrix
    :param class_names: class names
    :param normalize: normalize flag (default: True)
    :param save_path: save path (default: None for display only)
    """
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1, keepdims=True)

    num_classes = len(class_names)
    plt.figure()
    plt.imshow(conf_mat, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.show()
