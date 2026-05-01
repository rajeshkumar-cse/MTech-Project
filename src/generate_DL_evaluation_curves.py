import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
Evaluation Curves Generator
Generates ROC curves, Precision-Recall curves, and Calibration plots
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import config
from src.models import ECGXAINet
from src.dataset import PTBXLDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


def generate_DL_evaluation_curves():
    """
    Generate ROC curves, Precision-Recall curves, and Calibration plots
    """
    
    print("\n" + "="*60)
    print("Generating Evaluation Curves...")
    print("="*60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    if not os.path.exists(model_path):
        print("Model not found. Skipping evaluation curves.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']  # Added cyan for AFib
    
    # 1. ROC Curves
    print("Creating ROC curves...")
    plt.figure(figsize=(10, 8))
    
    for i in range(config.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                label=f'{config.DIAGNOSTIC_CLASSES[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves for Multi-Label Classification', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    os.makedirs(os.path.join(config.BASE_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'roc_curves.png'), dpi=300)
    plt.close()
    print("ROC curves saved to plots/roc_curves.png")
    
    # 2. Precision-Recall Curves
    print("\n" + "="*60)
    print("Creating Precision-Recall curves...")
    print("="*60)
    
    plt.figure(figsize=(10, 8))
    
    for i in range(config.NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])
        avg_precision = average_precision_score(all_labels[:, i], all_preds[:, i])
        
        plt.plot(recall, precision, color=colors[i], lw=2.5,
                label=f'{config.DIAGNOSTIC_CLASSES[i]} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Curves for Multi-Label Classification', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    os.makedirs(os.path.join(config.BASE_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'precision_recall_curves.png'), dpi=300)
    plt.close()
    print("Precision-Recall curves saved to plots/precision_recall_curves.png")
    
    # 3. Calibration Plots
    print("\n" + "="*60)
    print("Creating calibration plots...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(config.NUM_CLASSES):
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            all_labels[:, i], all_preds[:, i], n_bins=10, strategy='uniform'
        )
        
        # Plot calibration curve
        axes[i].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        axes[i].plot(mean_predicted_value, fraction_of_positives, 's-', 
                    color=colors[i], lw=2.5, markersize=8, label=f'{config.DIAGNOSTIC_CLASSES[i]}')
        
        axes[i].set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Fraction of Positives', fontsize=11, fontweight='bold')
        axes[i].set_title(f'{config.DIAGNOSTIC_CLASSES[i]}', fontsize=13, fontweight='bold')
        axes[i].legend(loc='upper left', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])
        axes[i].set_facecolor('#f8f9fa')
    
    # Hide extra subplot
    axes[5].axis('off')
    
    plt.suptitle('Calibration Curves for All Classes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.join(config.BASE_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'calibration_curves.png'), dpi=300)
    plt.close()
    print("Calibration curves saved to plots/calibration_curves.png")


if __name__ == '__main__':
    generate_DL_evaluation_curves()
