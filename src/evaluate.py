import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

def plot_roc_curve(labels, preds, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
        plt.plot(fpr, tpr, label=f'{name}')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(labels, preds, class_names, threshold=0.5, save_path=None):
    # Since it's multi-label, we can plot a confusion matrix for each class or an aggregated one
    # Here we plot a heatmap of co-occurrence or per-class metrics
    # For simplicity, let's just print the classification report
    from sklearn.metrics import classification_report
    preds_binary = (preds > threshold).astype(int)
    report = classification_report(labels, preds_binary, target_names=class_names, zero_division=0)
    print(report)
    
    # Save report to text file
    if save_path:
        with open(save_path.replace('.png', '.txt'), 'w') as f:
            f.write(report)
