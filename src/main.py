import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add MTP-2/ to path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PTBXLDataset
from models import ECGXAINet
from train import train_one_epoch, validate
from loss import FocalLoss
import config

def main():
    parser = argparse.ArgumentParser(description='ECG-XAI-Net Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode: train or eval')
    args = parser.parse_args()

    # Device
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA/GPU not available. Using CPU instead.")

    # Dataset & Dataloaders
    print("Loading data...")
    train_dataset = PTBXLDataset(config.DATA_DIR, split='train', sampling_rate=config.SAMPLING_RATE)
    val_dataset = PTBXLDataset(config.DATA_DIR, split='val', sampling_rate=config.SAMPLING_RATE)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    model = model.to(device)

    # Setup Training Components
    
    # Calculate Per-Class Weights for Imbalance
    from interpretability import compute_class_weights
    print("Computing per-class weights for rare class boosting...")
    class_weights = compute_class_weights(train_loader).to(device)
    print(f"Class Weights (NORM, MI, STTC, CD, HYP, AFib): {class_weights}")
    
    # Use Focal Loss with dynamic per-class weights
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Directories
    save_dir = os.path.join(config.BASE_DIR, 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')

    if args.mode == 'train':
        print("Starting training...")
        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(config.NUM_EPOCHS):
            # Enable Augmentation
            train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_augmentation=True)
            val_loss, val_auc, val_f1 = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            print("\n" + "="*80)
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} F1: {val_f1:.4f}")

            # Early Stopping and Checkpointing
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), save_path)
                print(f"⭐ Saved best model with AUC: {best_val_auc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    elif args.mode == 'eval':
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=device))
            print("Loaded best model.")
            
            test_dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
            
            test_loss, test_auc, test_f1 = validate(model, test_loader, criterion, device)
            print(f"Test Results | Loss: {test_loss:.4f} | AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
        else:
            print("No model checkpoint found to evaluate!")

if __name__ == '__main__':
    main()
