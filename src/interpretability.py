import torch
import torch
# import shap (Removed to use Captum)
import numpy as np
from captum.attr import GradientShap, IntegratedGradients

class ECGExplainer:
    def __init__(self, model, background_data, device):
        """
        Wrapper for SHAP explanations using Captum GradientShap.
        Args:
            model (nn.Module): Trained PyTorch model.
            background_data (torch.Tensor): Background dataset for SHAP (e.g., first 100 samples).
            device (torch.device): Device to run the model on.
        """
        self.model = model
        self.device = device
        self.background_data = background_data.to(device)
        self.explainer = GradientShap(self.model_forward_logits)

    def model_forward_logits(self, inputs):
        return self.model(inputs, return_logits=True)

    def explain(self, input_tensor):
        """
        Generate SHAP values for a given input.
        Args:
            input_tensor (torch.Tensor): Input sample (1, 12, 5000).
        Returns:
            list: SHAP values for each class output.
        """
        input_tensor = input_tensor.to(self.device).requires_grad_()
        
        # Calculate SHAP values for each target class
        shap_values_list = []
        # Assuming we want explanation for top predicted class or all?
        # Standard SHAP returns list for all outputs.
        # But GradientShap targets one output at a time.
        # To match previous behavior (list of arrays for each class), we loop.
        
        # However, computing everything is expensive.
        # Let's modify behavior slightly: Retrun SHAP for ALL classes?
        # Or just the top class?
        # The dashboard expects a LIST where index = class_idx.
        
        # To avoid computing 5 times (slow), we can just compute for the Predicted Class?
        # But the dashboard allows selecting "Target Class".
        # So explain() ideally should take target_class.
        # BUT the signature is explain(input_tensor).
        
        # I will change explain() signature later, but to keep compat with dashboard:
        # I will return a dummy list where only the Relevant indices are filled?
        # No, dashboard does: shap_vals[target_idx].
        
        # Okay, I'll loop over 5 classes. It might be slow.
        # Or, I can change dashboard to call explain(..., target=...)
        # But dashboard calls explainer.explain(input_tensor) generally.
        
        # Wait, previous SHAP explainer returned list of numpy arrays.
        # Captum returns tensor.
        
        # I will implement a lazy list or compute all 5. 5 is small.
        # num_classes in config is 5.
        
        import config
        num_classes = config.NUM_CLASSES
        shap_values_list = []
        
        # We need a baseline. background_data is used as baseline distribution.
        # GradientShap takes baselines=background_data.
        
        for i in range(num_classes):
            # Attribute for class i
            # n_samples determines how many background samples to use (randomly sampled?)
            # stdevs is for noise?
            # GradientShap approximates Expected Gradients.
            
            # baselines can be a tensor of multiple samples.
            # Captum averages over them.
            
            attr = self.explainer.attribute(input_tensor, baselines=self.background_data, target=i)
            shap_values_list.append(attr.detach().cpu().numpy())
            
        return shap_values_list

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output)
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        # Global Average Pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])
        
        # Weight the channels by gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU activation
        heatmap = torch.relu(heatmap)
        
        # Normalize
        heatmap /= torch.max(heatmap) + 1e-8
        
        return heatmap.cpu().numpy()

from captum.attr import IntegratedGradients

class BioInspiredExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(self.model_forward_logits)

    def model_forward_logits(self, inputs):
        return self.model(inputs, return_logits=True)

    def explain(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device).requires_grad_()
        # Baseline is zero
        baseline = torch.zeros_like(input_tensor)
        # n_steps=50 for good approximation
        attr = self.ig.attribute(input_tensor, baseline, target=target_class, n_steps=50)
        return attr.detach().cpu().numpy()

def get_attention_weights(model, input_tensor):
    """
    Extract attention weights from the Transformer encoder.
    Returns: Attention weights of the last layer (Batch, Heads, TargetSeq, SourceSeq)
    """
    # Forward pass to populate weights
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Access weights from the last layer of our custom encoder
    # Note: 'layers' is a ModuleList
    last_layer = model.transformer_encoder.layers[-1]
    
    if hasattr(last_layer, 'attn_weights') and last_layer.attn_weights is not None:
        return last_layer.attn_weights.detach().cpu().numpy()
    else:
        print("Warning: Attention weights not found. Ensure InterpretableTransformerEncoderLayer is used.")
        return None

class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) for ECG"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def explain(self, input_sample, num_samples=1000, num_features=10):
        """
        Generate LIME explanation for ECG signal
        Args:
            input_sample: (1, 12, 5000) ECG tensor
            num_samples: Number of perturbed samples for LIME
            num_features: Number of top features to show
        Returns:
            Feature importances for each lead
        """
        from lime import lime_tabular
        
        # Flatten input for LIME
        input_flat = input_sample.detach().cpu().numpy().reshape(1, -1)
        
        # Create prediction function
        def predict_fn(x):
            x_tensor = torch.FloatTensor(x).reshape(-1, 12, 5000).to(self.device)
            with torch.no_grad():
                preds = self.model(x_tensor).cpu().numpy()
            return preds
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            input_flat,
            mode='regression',
            feature_names=[f'Lead{i//5000+1}_T{i%5000}' for i in range(12*5000)]
        )
        
        # Explain
        exp = explainer.explain_instance(
            input_flat[0],
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return exp

def compute_per_lead_importance(shap_values, signal_shape=(12, 5000)):
    """
    Compute importance score for each ECG lead from SHAP values
    Args:
        shap_values: SHAP attribution values - can be (12,) or (12, 5000) - can be tensor or numpy array
        signal_shape: Shape of ECG signal
    Returns:
        lead_importance: (12,) array of importance scores per lead
    """
    import torch
    
    if isinstance(shap_values, list):
        # If multiple outputs, use first class
        shap_values = shap_values[0]
    
    # Convert tensor to numpy if needed
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()
    
    # Reshape if needed
    if len(shap_values.shape) == 3:
        shap_values = shap_values[0]  # Remove batch dim
    
    # Check if already aggregated (shape is (12,)) or needs aggregation (shape is (12, 5000))
    if len(shap_values.shape) == 1:
        # Already aggregated per lead
        lead_importance = np.abs(shap_values)
    elif len(shap_values.shape) == 2:
        # Compute absolute mean importance per lead
        lead_importance = np.abs(shap_values).mean(axis=1)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    
    return lead_importance

def compute_class_weights(train_loader):
    """
    Compute class weights for imbalanced dataset (FAST VERSION)
    Args:
        train_loader: DataLoader for training data
    Returns:
        pos_weight: Tensor of shape (num_classes,) for weighting rare classes
    """
    # Access dataset directly instead of iterating through loader (100x faster!)
    dataset = train_loader.dataset
    all_labels = dataset.labels  # This is already a numpy array
    
    # Count positive samples per class
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(all_labels) - pos_counts
    
    # Compute weights (neg/pos ratio)
    pos_weight = neg_counts / (pos_counts + 1e-8)
    
    return torch.FloatTensor(pos_weight)

