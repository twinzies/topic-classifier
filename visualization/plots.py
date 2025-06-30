"""
Description: Visualization utilities for metrics and results.
"""

import logging

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_loss_curve(loss_values):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    logger.info("Loss curve plotted successfully.")

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    logger.info("Confusion matrix plotted successfully.")
