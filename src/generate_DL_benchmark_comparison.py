import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
Benchmark Comparison Generator
Runs benchmark comparison against SOTA baselines
"""

import matplotlib.pyplot as plt
import torch
import os
import config
from src.models import ECGXAINet
from src.dataset import PTBXLDataset
from src.train import validate


def generate_DL_benchmark_comparison():
    """Run benchmark comparison against baselines"""
    print("\n" + "="*60)
    print("Running Benchmark Comparison...")
    print("="*60)
    
    # Baselines from Report
    baselines = {
        'Wagner et al. (2020) [Baseline]': 0.937,
        'Ayano et al. (2024) [Hybrid]': 0.928,
        'Our Model (ECG-XAI-Net)': 0.0 # To be filled
    }
    
    # Load our model and evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    
    if os.path.exists(model_path):
        # weights_only=True suppresses the warning and is safer
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        
        test_dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        print("Evaluating model on test set for benchmarking...")
        # Use Focal Loss if that's what we trained with, or just BCE for validation metric
        # Validation function returns AUC which is loss-agnostic
        from src.loss import FocalLoss
        criterion = FocalLoss() 
        test_loss, auc, f1 = validate(model, test_loader, criterion, device)
        
        # Check if we hit the target!
        print(f"✅ Our Model Final AUC: {auc:.4f}")
        baselines['Our Model (ECG-XAI-Net)'] = auc
    else:
        print("Model checkpoint not found. Using placeholder 0.0 for Our Model.")

    # Plot Comparison - Beautiful Bar Chart
    names = list(baselines.keys())
    values = list(baselines.values())
    
    # Define beautiful colors for each bar
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    
    plt.figure(figsize=(12, 7))
    
    # Create bar chart
    bars = plt.bar(names, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add gradient effect by adjusting alpha
    for i, bar in enumerate(bars):
        bar.set_alpha(0.9 if i == 2 else 0.75)  # Highlight our model
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        label_text = '{:.4f}'.format(val)
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                label_text,
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Styling
    plt.ylim(0.80, 1.0)  # Better range to show differences
    plt.ylabel('Macro AUC-ROC Score', fontsize=14, fontweight='bold')
    plt.xlabel('Models', fontsize=14, fontweight='bold')
    plt.title('Performance Comparison: ECG-XAI-Net vs State-of-the-Art', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add horizontal grid for easier reading
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=15, ha='right')
    
    # Add a subtle background color
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    save_path = os.path.join(config.BASE_DIR, 'plots', 'benchmark_comparison_graph.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Benchmark graph saved to {save_path}")


if __name__ == '__main__':
    generate_DL_benchmark_comparison()
