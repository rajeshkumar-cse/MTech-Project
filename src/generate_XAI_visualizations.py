import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
XAI Visualizations Generator
Generates SHAP, Grad-CAM, Integrated Gradients, and Attention Map visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from src.models import ECGXAINet
from src.dataset import PTBXLDataset
from src.interpretability import ECGExplainer, BioInspiredExplainer, GradCAM, get_attention_weights
from scipy.interpolate import interp1d
import traceback


def generate_XAI_visualizations():
    """
    Generate comprehensive XAI visualizations:
    1. SHAP
    2. Grad-CAM
    3. Integrated Gradients
    4. Attention Maps
    """
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load a few test samples
    test_dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE)
    # Get a batch
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True)
    inputs, labels = next(iter(loader))
    inputs = inputs.to(device) # (5, 12, 5000)

    # 1. Prediction
    with torch.no_grad():
        preds = model(inputs)
    
    print("Predictions:\n", preds.cpu().numpy())
    print("True Labels:\n", labels.cpu().numpy())

    # 2. Generate SHAP Values
    # Use a background set (e.g., first 50 training samples)
    train_dataset = PTBXLDataset(config.DATA_DIR, split='train', sampling_rate=config.SAMPLING_RATE)
    background_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
    background_data, _ = next(iter(background_loader))
    
    explainer = ECGExplainer(model, background_data, device)
    
    # Explain the first test sample
    # Note: DeepExplainer can be memory intensive. Explain a small batch or single sample.
    sample_idx = 0
    input_sample = inputs[sample_idx:sample_idx+1] # (1, 12, 5000)
    
    print("\n" + "="*60)
    print("Generating SHAP values...")
    print("="*60)
    try:
        shap_values = explainer.explain(input_sample)
    except Exception as e:
        print(f"Warning: SHAP generation failed: {e}")
        print("Falling back to plotting only the raw signal in 3D.")
        shap_values = None # Signal to skip coloring
    
    # 3. Plot SHAP (Heatmap for 12 leads)
    
    # Create directory for plots
    os.makedirs(os.path.join(config.BASE_DIR, 'plots'), exist_ok=True)
    
    # Example: Plot Lead II (index 1) for Class 'MI' (index 1)
    target_class_idx = 1 # MI
    lead_idx = 1 # Lead II
    
    # If SHAP failed, use zeros
    original_signal = input_sample[0, lead_idx, :].detach().cpu().numpy()
    if shap_values is None:
        explanation = np.zeros_like(original_signal)
    else:
        if isinstance(shap_values, list):
            # GradientExplainer returns list of tensors
            shap_attr = shap_values[target_class_idx] # (1, 12, 5000)
        else:
            shap_attr = shap_values # If not list
            
        # Check shape safety
        try:
             explanation = shap_attr[0, lead_idx, :] # (5000,)
             if explanation.shape != original_signal.shape:
                 print(f"Shape mismatch: explanation {explanation.shape} vs signal {original_signal.shape}. Using zeros.")
                 explanation = np.zeros_like(original_signal)
        except Exception:
             explanation = np.zeros_like(original_signal)
    
    plt.figure(figsize=(12, 4))
    plt.plot(original_signal, label='Original signal (Lead II)', alpha=0.7)
    # Overlay SHAP as heatmap or color
    # Normalize explanation for color map
    if np.max(np.abs(explanation)) > 1e-8:
        norm_explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation) + 1e-8)
    else:
        norm_explanation = np.zeros_like(explanation)
    
    # Scatter plot with color
    tk = np.arange(len(original_signal))
    plt.scatter(tk, original_signal, c=explanation, cmap='coolwarm', s=1, alpha=0.8, label='SHAP Importance')
    plt.colorbar(label='SHAP Value')
    plt.title(f'SHAP Explanation for Class {config.DIAGNOSTIC_CLASSES[target_class_idx]} (Lead II)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'shap_example.png'))
    plt.close()
    
    print("SHAP plot saved to plots/shap_example.png")

    # 4. Grad-CAM Visualization
    print("\n" + "="*60)
    print("Generating Grad-CAM...")
    print("="*60)
    
    # Target the last convolutional layer
    # Note: In ECGXAINet, model.conv4 is the last conv layer
    grad_cam = GradCAM(model, model.conv4)
    
    try:
        # We need to ensure input requires grad for backward pass in GradCAM
        input_sample.requires_grad = True
        cam = grad_cam.generate_cam(input_sample, target_class=target_class_idx)
        
        # Upsample CAM to match signal length (313 -> 5000)
        # cam shape is (313,) roughly
        current_len = len(cam)
        target_len = input_sample.shape[2]
        
        # Interpolate
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_len)
        f = interp1d(x_old, cam, kind='linear')
        cam_upsampled = f(x_new)
        
        # Plot Grad-CAM
        plt.figure(figsize=(12, 4))
        plt.plot(original_signal, label='Original Signal (Lead II)', alpha=0.7)
        # Normalize original signal for plotting CAM on top
        max_val = np.max(np.abs(original_signal)) if np.max(np.abs(original_signal)) > 0 else 1.0
        
        plt.plot(cam_upsampled * max_val, label='Grad-CAM', color='red', linewidth=2)
        plt.fill_between(range(target_len), 0, cam_upsampled * max_val, color='red', alpha=0.3)
        
        plt.title(f'Grad-CAM Visualization for Class {config.DIAGNOSTIC_CLASSES[target_class_idx]} (Lead II)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'grad_cam_example.png'))
        plt.close()
        print("Grad-CAM plot saved to plots/grad_cam_example.png")
        
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")

    # 5. Integrated Gradients Visualization (DeepLIFT Equivalent)
    print("\n" + "="*60)
    print("Generating Integrated Gradients attributions...")
    print("="*60)
    try:
        ig_explainer = BioInspiredExplainer(model, device)
        # DeepLift typically requires baseline. We use zero baseline internally.
        ig_attr = ig_explainer.explain(input_sample, target_class=target_class_idx)
        print(f"IG attr shape: {ig_attr.shape}")
        
        # ig_attr shape: (1, 12, 5000)
        ig_explanation = ig_attr[0, lead_idx, :]
        
        plt.figure(figsize=(12, 4))
        plt.plot(original_signal, label='Original signal', alpha=0.7)
        
        # Create a colormap
        # Use abs max for symmetric colorbar if data is centered, or just min/max
        norm_ig = plt.Normalize(vmin=np.min(ig_explanation), vmax=np.max(ig_explanation))
        plt.scatter(tk, original_signal, c=ig_explanation, cmap='PRGn', s=1, alpha=0.9, label='IG Attribution')
        plt.colorbar(label='Attribution')
        plt.title(f'Integrated Gradients (DeepLIFT) for Class {config.DIAGNOSTIC_CLASSES[target_class_idx]} (Lead II)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'deeplift_example.png')) # Keep filename for consistency
        plt.close()
        print("Integrated Gradients plot saved to plots/deeplift_example.png")
    except Exception as e:
        print(f"IG generation failed: {e}")
        traceback.print_exc()

    # 6. Attention Map Visualization
    print("\n" + "="*60)
    print("Generating Attention Map...")
    print("="*60)
    try:
        # Get attention weights from the last Transformer layer
        attn_weights = get_attention_weights(model, input_sample) # Expect (1, L, L) or (1, Heads, L, L)
        
        if attn_weights is not None:
            print(f"Attention weights shape: {attn_weights.shape}")
            
            # Handle shapes
            if len(attn_weights.shape) == 3:
                # (Batch, L, L) - Average over heads was already done by MultiheadAttention
                avg_attn = attn_weights[0] # (L, L)
            elif len(attn_weights.shape) == 4:
                # (Batch, Heads, L, L)
                avg_attn = np.mean(attn_weights[0], axis=0) # (L, L)
            else:
                raise ValueError(f"Unexpected attention shape: {attn_weights.shape}")

            # To get "Time Importance", we can average over the Query dimension (axis 0 of the L,L matrix)
            # This shows which Source tokens (cols) were attended to by all Queries (rows)
            time_importance = np.mean(avg_attn, axis=0) # (L,)
            
            # Upsample to original length
            current_len = len(time_importance)
            target_len = input_sample.shape[2]
            f = interp1d(np.linspace(0, 1, current_len), time_importance, kind='linear')
            attn_upsampled = f(np.linspace(0, 1, target_len))
            
            # Normalize for plotting
            attn_upsampled = (attn_upsampled - np.min(attn_upsampled)) / (np.max(attn_upsampled) - np.min(attn_upsampled) + 1e-8)
            
            plt.figure(figsize=(12, 4))
            plt.plot(original_signal, label='Original Signal', alpha=0.7)
            # Fill under the curve with attention importance
            plt.fill_between(range(target_len), np.min(original_signal), np.max(original_signal), 
                             where=attn_upsampled > 0.5, color='orange', alpha=0.3, label='High Attention')
            
            # Also plot the attention curve overlaid
            # Scale to signal amplitude range for visibility
            amp_scale = np.max(np.abs(original_signal)) if np.max(np.abs(original_signal)) > 0 else 1.0
            plt.plot(attn_upsampled * amp_scale, label='Attention Weights', color='orange', linewidth=2)
            
            plt.title(f'Transformer Attention Map (Global Average) for Class {config.DIAGNOSTIC_CLASSES[target_class_idx]} (Lead II)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(config.BASE_DIR, 'plots', 'attention_map_example.png'))
            plt.close()
            print("Attention Map saved to plots/attention_map_example.png")
            
    except Exception as e:
        print(f"Attention Map generation failed: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    generate_XAI_visualizations()
