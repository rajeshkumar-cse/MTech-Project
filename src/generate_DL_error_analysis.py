import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
Automated Error Analysis Module
Performs comprehensive model diagnostics and generates insights
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import config
from src.models import ECGXAINet
from src.dataset import PTBXLDataset
from torch.utils.data import DataLoader
from sklearn.metrics import multilabel_confusion_matrix


def generate_DL_error_analysis():
    """
    Perform automated error analysis on model predictions
    Generates comprehensive diagnostic plots and insights
    """
    
    print("\n" + "="*60)
    print("Performing Automated Error Analysis...")
    print("="*60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    if not os.path.exists(model_path):
        print("Model not found. Skipping error analysis.")
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
    
    # Analyze misclassifications
    pred_binary = (all_preds > 0.5).astype(int)
    
    # 1. Class-wise confusion analysis
    confusion_matrices = multilabel_confusion_matrix(all_labels, pred_binary)
    
    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(config.DIAGNOSTIC_CLASSES):
        tn, fp, fn, tp = confusion_matrices[i].ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    # 2. Confidence analysis for correct vs incorrect predictions
    confidence_correct = []
    confidence_incorrect = []
    
    for i in range(len(all_preds)):
        for j in range(config.NUM_CLASSES):
            pred_prob = all_preds[i, j]
            true_label = all_labels[i, j]
            pred_label = pred_binary[i, j]
            
            if pred_label == true_label:
                confidence_correct.append(pred_prob if pred_label == 1 else 1 - pred_prob)
            else:
                confidence_incorrect.append(pred_prob if pred_label == 1 else 1 - pred_prob)
    
    # 3. Find most confused class pairs
    confusion_pairs = []
    for i in range(config.NUM_CLASSES):
        for j in range(i + 1, config.NUM_CLASSES):
            # Count cases where both are predicted but only one is true
            both_pred = (pred_binary[:, i] == 1) & (pred_binary[:, j] == 1)
            only_i_true = (all_labels[:, i] == 1) & (all_labels[:, j] == 0)
            only_j_true = (all_labels[:, i] == 0) & (all_labels[:, j] == 1)
            
            confusion_count = np.sum(both_pred & (only_i_true | only_j_true))
            if confusion_count > 0:
                confusion_pairs.append((
                    config.DIAGNOSTIC_CLASSES[i],
                    config.DIAGNOSTIC_CLASSES[j],
                    confusion_count
                ))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Generate visualizations
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Per-class performance metrics
    ax1 = plt.subplot(2, 3, 1)
    metrics_data = np.array([[class_metrics[c]['precision'], 
                              class_metrics[c]['recall'], 
                              class_metrics[c]['f1']] 
                             for c in config.DIAGNOSTIC_CLASSES])
    
    x = np.arange(len(config.DIAGNOSTIC_CLASSES))
    width = 0.25
    
    ax1.bar(x - width, metrics_data[:, 0], width, label='Precision', color='#3498db', alpha=0.8)
    ax1.bar(x, metrics_data[:, 1], width, label='Recall', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width, metrics_data[:, 2], width, label='F1-Score', color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('Disease Class', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Per-Class Performance Metrics', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(config.DIAGNOSTIC_CLASSES, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Confidence distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(confidence_correct, bins=20, alpha=0.7, label='Correct Predictions', 
             color='#2ecc71', edgecolor='black')
    ax2.hist(confidence_incorrect, bins=20, alpha=0.7, label='Incorrect Predictions', 
             color='#e74c3c', edgecolor='black')
    ax2.set_xlabel('Prediction Confidence', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Confidence Distribution Analysis', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: False Positive vs False Negative rates
    ax3 = plt.subplot(2, 3, 3)
    fp_rates = [class_metrics[c]['fp'] / (class_metrics[c]['fp'] + class_metrics[c]['tn']) 
                if (class_metrics[c]['fp'] + class_metrics[c]['tn']) > 0 else 0 
                for c in config.DIAGNOSTIC_CLASSES]
    fn_rates = [class_metrics[c]['fn'] / (class_metrics[c]['fn'] + class_metrics[c]['tp']) 
                if (class_metrics[c]['fn'] + class_metrics[c]['tp']) > 0 else 0 
                for c in config.DIAGNOSTIC_CLASSES]
    
    x = np.arange(len(config.DIAGNOSTIC_CLASSES))
    ax3.bar(x - 0.2, fp_rates, 0.4, label='False Positive Rate', color='#e74c3c', alpha=0.8)
    ax3.bar(x + 0.2, fn_rates, 0.4, label='False Negative Rate', color='#f39c12', alpha=0.8)
    ax3.set_xlabel('Disease Class', fontweight='bold')
    ax3.set_ylabel('Error Rate', fontweight='bold')
    ax3.set_title('Error Type Analysis', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(config.DIAGNOSTIC_CLASSES, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Most confused class pairs
    ax4 = plt.subplot(2, 3, 4)
    if confusion_pairs:
        top_pairs = confusion_pairs[:5]
        pair_labels = [f"{p[0]}\n↔\n{p[1]}" for p in top_pairs]
        pair_counts = [p[2] for p in top_pairs]
        
        bars = ax4.barh(pair_labels, pair_counts, color='#9b59b6', alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Confusion Count', fontweight='bold')
        ax4.set_title('Most Confused Class Pairs', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, count in zip(bars, pair_counts):
            width = bar.get_width()
            ax4.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{int(count)}', ha='left', va='center', fontweight='bold')
    
    # Plot 5: Sample size distribution
    ax5 = plt.subplot(2, 3, 5)
    positive_counts = [int(class_metrics[c]['tp'] + class_metrics[c]['fn']) 
                       for c in config.DIAGNOSTIC_CLASSES]
    bars = ax5.bar(config.DIAGNOSTIC_CLASSES, positive_counts, color='#1abc9c', alpha=0.8, edgecolor='black')
    ax5.set_xlabel('Disease Class', fontweight='bold')
    ax5.set_ylabel('Number of Positive Samples', fontweight='bold')
    ax5.set_title('Class Distribution in Test Set', fontweight='bold', fontsize=12)
    ax5.set_xticks(range(len(config.DIAGNOSTIC_CLASSES)))
    ax5.set_xticklabels(config.DIAGNOSTIC_CLASSES, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, count in zip(bars, positive_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Auto-generated insights (text summary)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Generate insights
    best_class = max(class_metrics.items(), key=lambda x: x[1]['f1'])
    worst_class = min(class_metrics.items(), key=lambda x: x[1]['f1'])
    avg_confidence_correct = np.mean(confidence_correct) if confidence_correct else 0
    avg_confidence_incorrect = np.mean(confidence_incorrect) if confidence_incorrect else 0
    
    insights_text = f"""
    AUTOMATED INSIGHTS
    
    [+] Best Performance:
       {best_class[0]} (F1: {best_class[1]['f1']:.3f})
    
    [!] Needs Improvement:
       {worst_class[0]} (F1: {worst_class[1]['f1']:.3f})
    
    [*] Confidence Analysis:
       Correct: {avg_confidence_correct:.1%}
       Incorrect: {avg_confidence_incorrect:.1%}
    
    [~] Top Confusion:
       {confusion_pairs[0][0]} <-> {confusion_pairs[0][1]}
       ({confusion_pairs[0][2]} cases)
    
    [>] Recommendation:
       {"Model shows good calibration" if avg_confidence_correct > avg_confidence_incorrect + 0.1 
        else "Consider confidence calibration"}
    """
    
    ax6.text(0.1, 0.9, insights_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Automated Error Analysis & Model Diagnostics', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.join(config.BASE_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Error analysis saved to plots/error_analysis.png")
    
    # Print insights to console
    print("\n📊 KEY INSIGHTS:")
    print(f"✅ Best performing class: {best_class[0]} (F1: {best_class[1]['f1']:.3f})")
    print(f"⚠️  Weakest class: {worst_class[0]} (F1: {worst_class[1]['f1']:.3f})")
    print(f"🎯 Avg confidence (correct): {avg_confidence_correct:.1%}")
    print(f"🎯 Avg confidence (incorrect): {avg_confidence_incorrect:.1%}")
    if confusion_pairs:
        print(f"🔄 Most confused pair: {confusion_pairs[0][0]} ↔ {confusion_pairs[0][1]} ({confusion_pairs[0][2]} cases)")


if __name__ == '__main__':
    generate_DL_error_analysis()
