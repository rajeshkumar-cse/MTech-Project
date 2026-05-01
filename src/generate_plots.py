import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import ECGXAINet
from src.dataset import PTBXLDataset
from src.interpretability import ECGExplainer
from src.train import validate
import config
import os

def generate_XAI_visualizations():
    """Generate comprehensive XAI visualizations (SHAP, Grad-CAM, etc.)"""
    from generate_XAI_visualizations import generate_XAI_visualizations
    generate_XAI_visualizations()



def generate_CV_Visualization_report():
    """Generate computer vision style evaluation plots"""
    from generate_CV_Visualization_report import generate_CV_Visualization_report
    generate_CV_Visualization_report()


def generate_interactive_3d():
    """Generate interactive matplotlib 3D ECG visualization in a window"""
    from torch.utils.data import DataLoader
    from mpl_toolkits.mplot3d import Axes3D
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    if not os.path.exists(model_path):
        print("Model not found. Skipping interactive 3D.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Load test sample
    test_dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    inputs, _ = next(iter(loader))
    
    signal = inputs[0].cpu().numpy()  # (12, 5000)
    
    # Downsample for performance
    step = 10
    signal_ds = signal[:, ::step]
    
    # Define vibrant colors for each lead
    lead_colors = [
        '#e74c3c',  # Red
        '#3498db',  # Blue
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#e67e22',  # Carrot
        '#34495e',  # Dark Gray
        '#f1c40f',  # Yellow
        '#e91e63',  # Pink
        '#00bcd4',  # Cyan
        '#ff5722'   # Deep Orange
    ]
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each lead
    for i in range(12):
        z = signal_ds[i, :]
        x = np.arange(0, signal.shape[1], step)
        y = np.full_like(x, i * 10)
        
        ax.plot(x, y, z, color=lead_colors[i], linewidth=2, label=lead_names[i])
    
    # Customize plot
    ax.set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lead Index', fontsize=12, fontweight='bold')
    ax.set_zlabel('Amplitude (mV)', fontsize=12, fontweight='bold')
    ax.set_title('Interactive 3D 12-Lead ECG Visualization', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    print("\n" + "="*60)
    print("🎨 3D ECG Viewer Opened!")
    print("="*60)
    print("\n📌 Controls:")
    print("  • Click and drag to rotate")
    print("  • Scroll to zoom")
    print("  • Right-click drag to pan")
    print("  • Close window when done")
    print("\n" + "="*60 + "\n")
    
    # Show interactive window
    plt.show()


def generate_XAI_lime_and_lead_importance():
    """Generate LIME explanation and per-lead importance plots"""
    from generate_XAI_lime_explanation import generate_XAI_lime_explanation
    generate_XAI_lime_explanation()




def run_benchmark():
    """Run benchmark comparison against baselines"""
    from generate_DL_benchmark_comparison import generate_DL_benchmark_comparison
    generate_DL_benchmark_comparison()

if __name__ == '__main__':
    generate_XAI_visualizations()
    generate_CV_Visualization_report()
    run_benchmark()
    
    # Generate performance comparison table
    print("\n" + "="*60)
    print("Generating Performance Comparison Table...")
    print("="*60)
    from generate_performance_comparison import create_performance_comparison
    create_performance_comparison()
    
    generate_interactive_3d()
    generate_XAI_lime_and_lead_importance()
    # PDF report generation removed as per user request
    
    # Generate clinical reports
    from generate_GenAI_clinical_report import generate_GenAI_sample_reports
    generate_GenAI_sample_reports(num_samples=3)



