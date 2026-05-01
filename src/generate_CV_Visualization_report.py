import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
CV Style Visualization Report Generator
Generates confusion matrices and triggers other evaluation plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
import config
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from src.models import ECGXAINet
from src.dataset import PTBXLDataset

# Import other evaluation modules
from generate_DL_evaluation_curves import generate_DL_evaluation_curves
from generate_DL_error_analysis import generate_DL_error_analysis


def generate_CV_Visualization_report():
    """
    Generate computer vision style evaluation plots
    Includes Confusion Matrices, ROC/PR Curves, and Error Analysis
    """
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    if not os.path.exists(model_path):
        print("Model not found. Skipping CV visualizations.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE)
    # Use larger batch size for inference
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_labels = []
    
    print("\n" + "="*60)
    print("Generating predictions for CV visualizations...")
    print("="*60)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 1. Confusion Matrix (for each class)
    print("Creating confusion matrix...")
    pred_binary = (all_preds > 0.5).astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(config.NUM_CLASSES):
        cm = confusion_matrix(all_labels[:, i], pred_binary[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[i].set_title(f'{config.DIAGNOSTIC_CLASSES[i]}', fontweight='bold')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    # Hide extra subplot
    axes[5].axis('off')
    
    plt.suptitle('Confusion Matrices for All Classes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(config.BASE_DIR, 'plots', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    # 2-4. ROC, Precision-Recall, and Calibration Curves
    # Delegated to specific module
    generate_DL_evaluation_curves()
    
    # 5. Automated Error Analysis
    # Delegated to specific module
    generate_DL_error_analysis()


if __name__ == '__main__':
    generate_CV_Visualization_report()
