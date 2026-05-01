import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for addressing class imbalance.
        Args:
            alpha: Weighting factor for rare class. Can be:
                   - float: same weight for all classes
                   - torch.Tensor: per-class weights (shape: [num_classes])
            gamma: Focusing parameter for hard examples (default: 2)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Predictions (probabilities after sigmoid), shape: [batch, num_classes]
        targets: Ground truth labels (0 or 1), shape: [batch, num_classes]
        """
        # Compute BCE loss per sample and class
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Handle per-class alpha weights
        if isinstance(self.alpha, torch.Tensor):
            # alpha is a tensor of shape [num_classes]
            # Broadcast to [batch, num_classes]
            alpha_t = self.alpha.unsqueeze(0)  # [1, num_classes]
            F_loss = alpha_t * focal_term * BCE_loss
        else:
            # alpha is a scalar
            F_loss = self.alpha * focal_term * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
