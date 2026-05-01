import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Keep (MaxLen, 1, Dim) for compatibility
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        # pe: (MaxLen, 1, Dim)
        # Slice pe to (Seq, 1, Dim) then permute to (1, Seq, Dim)
        pe_slice = self.pe[:x.size(1), :].permute(1, 0, 2)
        return x + pe_slice

class InterpretableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    # Override for PyTorch 2.x compatibility
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # We force need_weights=True to capture them for visualization
        x, weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           is_causal=is_causal)
        self.attn_weights = weights # (Batch, Heads, TargetSeq, SourceSeq)
        return self.dropout1(x)

class ECGXAINet(nn.Module):
    def __init__(self, num_leads=12, num_classes=5, hidden_dim=512, nhead=8, num_layers=4):
        super(ECGXAINet, self).__init__()
        
        # 1. Feature Extraction (CNN Part)
        # Input: (Batch, 12, 5000)
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(2) # Downsample x2

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Skip connection handling: add projection if channels mismatch
        self.skip_proj1 = nn.Conv1d(num_leads, 64, kernel_size=1)
        self.skip_proj2 = nn.Conv1d(64, 128, kernel_size=1)
        
        # 2. Transformer Encoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        # Use our custom layer to capture attention weights
        encoder_layer = InterpretableTransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=2048, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. Classification Head
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x, return_logits=False):
        # x shape: (Batch, 12, 5000)
        
        # CNN Feature Extraction
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = self.relu(self.bn4(self.conv4(x))) # Final Conv Block
        # Output shape: (Batch, 512, Length')
        
        # Prepare for Transformer: (Batch, Time, Channels)
        x = x.permute(0, 2, 1) 
        
        # Positional Encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Global Pooling (over time dimension)
        # x back to (Batch, Channels, Time) for pooling
        x = x.permute(0, 2, 1)
        x = self.global_pool(x)
        x = x.squeeze(-1) # (Batch, 512)
        
        # Classification Head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        if return_logits:
            return logits
            
        return torch.sigmoid(logits) # Multi-label probabilities
