import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device, use_augmentation=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply augmentation if enabled
        # Apply augmentation if enabled
        if use_augmentation:
            from src.augmentation import ECGAugmentation
            inputs = torch.stack([ECGAugmentation.apply_random_augmentation(inp) for inp in inputs])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Check for NaN/Inf in predictions
    if np.isnan(all_preds).any() or np.isinf(all_preds).any():
        print("Warning: NaN or Inf found in predictions")
        all_preds = np.nan_to_num(all_preds)

    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auc = 0.5 # Default if only one class present in batch

    return epoch_loss, auc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auc = 0.5

    # Calculate F1-score (threshold=0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_binary, average='macro')

    return epoch_loss, auc, f1

def predict_with_uncertainty(model, dataloader, device, n_iterations=10):
    """
    Monte Carlo Dropout for uncertainty quantification
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device
        n_iterations: Number of MC dropout iterations
    Returns:
        mean_preds: Mean predictions
        std_preds: Standard deviation (uncertainty)
    """
    model.train()  # Enable dropout
    all_predictions = []
    
    for _ in range(n_iterations):
        preds = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds.append(outputs.cpu().numpy())
        all_predictions.append(np.concatenate(preds))
    
    all_predictions = np.array(all_predictions)
    mean_preds = all_predictions.mean(axis=0)
    std_preds = all_predictions.std(axis=0)
    
    return mean_preds, std_preds

