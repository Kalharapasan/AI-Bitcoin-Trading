"""
CNN-LSTM Hybrid Models for Time Series Forecasting
Combines convolutional neural networks for spatial feature extraction 
with LSTMs for temporal modeling in financial time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import AdamW, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import stats
import math

warnings.filterwarnings('ignore')

# ============ CNN Modules ============
class TemporalCNN(nn.Module):
    """Temporal CNN for feature extraction from time series"""
    
    def __init__(self, input_dim: int, filters: List[int], kernel_sizes: List[int],
                 pool_sizes: List[int] = None, dropout: float = 0.3, 
                 use_batch_norm: bool = True, dilation: bool = False):
        super().__init__()
        
        assert len(filters) == len(kernel_sizes), "Filters and kernel_sizes must have same length"
        
        if pool_sizes is None:
            pool_sizes = [2] * len(filters)
        
        layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
            # Convolutional layer
            if dilation and i > 0:
                dilation_rate = 2 ** i
                padding = ((kernel_size - 1) * dilation_rate) // 2
                conv_layer = nn.Conv1d(
                    in_channels, out_channels, kernel_size, 
                    padding=padding, dilation=dilation_rate,
                    padding_mode='replicate'
                )
            else:
                padding = kernel_size // 2
                conv_layer = nn.Conv1d(
                    in_channels, out_channels, kernel_size, 
                    padding=padding, padding_mode='replicate'
                )
            
            layers.append(conv_layer)
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            
            # Activation
            layers.append(nn.GELU())
            
            # Pooling (except last layer)
            if i < len(filters) - 1:
                layers.append(nn.MaxPool1d(pool_size))
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        self.output_channels = filters[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            CNN features of shape (batch_size, new_seq_len, output_channels)
        """
        # CNN expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        features = self.cnn(x)
        # Transpose back to [batch, seq_len, channels]
        return features.transpose(1, 2)

class ResidualCNNBlock(nn.Module):
    """Residual block for CNN with skip connections"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=kernel_size//2, padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                              padding=kernel_size//2, padding_mode='replicate')
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Skip connection
        out = self.activation(out)
        return out

class MultiScaleCNN(nn.Module):
    """Multi-scale CNN capturing features at different temporal resolutions"""
    
    def __init__(self, input_dim: int, base_filters: int = 64, dropout: float = 0.3):
        super().__init__()
        
        # Branch 1: Small kernel for short-term patterns
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_dim, base_filters, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Conv1d(base_filters, base_filters, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Branch 2: Medium kernel for medium-term patterns
        self.branch2 = nn.Sequential(
            nn.Conv1d(input_dim, base_filters, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Conv1d(base_filters, base_filters, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Branch 3: Large kernel for long-term patterns
        self.branch3 = nn.Sequential(
            nn.Conv1d(input_dim, base_filters, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Conv1d(base_filters, base_filters, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(base_filters * 3, base_filters, kernel_size=1),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, seq_len]
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        
        # Concatenate along channel dimension
        combined = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        fused = self.fusion(combined)
        
        return fused

# ============ LSTM Modules ============
class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Set forget gate bias
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Context vector (batch_size, hidden_dim)
            attention_weights: Attention weights (batch_size, seq_len) if return_attention=True
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Compute attention scores
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        if return_attention:
            return context_vector, attention_weights
        return context_vector

# ============ CNN-LSTM Hybrid Models ============
class CNNLSTMModel(nn.Module):
    """Standard CNN-LSTM hybrid model"""
    
    def __init__(self, input_dim: int, cnn_filters: List[int], cnn_kernel_sizes: List[int],
                 lstm_hidden: int, lstm_layers: int, output_dim: int, dropout: float = 0.3,
                 use_attention: bool = True, bidirectional_lstm: bool = True):
        super().__init__()
        
        # CNN for feature extraction
        self.cnn = TemporalCNN(
            input_dim=input_dim,
            filters=cnn_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            use_batch_norm=True
        )
        
        # Calculate CNN output dimension (sequence length may be reduced)
        cnn_output_dim = cnn_filters[-1]
        
        # LSTM for temporal modeling
        self.use_attention = use_attention
        if use_attention:
            self.lstm = AttentionLSTM(
                input_dim=cnn_output_dim,
                hidden_dim=lstm_hidden,
                num_layers=lstm_layers,
                bidirectional=bidirectional_lstm,
                dropout=dropout
            )
            lstm_output_dim = lstm_hidden * (2 if bidirectional_lstm else 1)
        else:
            self.lstm = nn.LSTM(
                input_size=cnn_output_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=bidirectional_lstm
            )
            lstm_output_dim = lstm_hidden * (2 if bidirectional_lstm else 1)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Forward pass through CNN-LSTM model
        """
        # CNN feature extraction
        cnn_features = self.cnn(x)
        cnn_features = self.dropout(cnn_features)
        
        # LSTM temporal modeling
        if self.use_attention:
            if return_attention:
                lstm_output, attention_weights = self.lstm(cnn_features, return_attention=True)
            else:
                lstm_output = self.lstm(cnn_features, return_attention=False)
        else:
            lstm_output, _ = self.lstm(cnn_features)
            lstm_output = lstm_output[:, -1, :]  # Take last timestep
            attention_weights = None
        
        # Output layer
        output = self.output_layer(lstm_output)
        
        if return_attention and self.use_attention:
            return output, attention_weights
        return output

class MultiScaleCNNLSTM(nn.Module):
    """Multi-scale CNN with LSTM"""
    
    def __init__(self, input_dim: int, cnn_base_filters: int, lstm_hidden: int,
                 lstm_layers: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Multi-scale CNN
        self.cnn = MultiScaleCNN(input_dim, cnn_base_filters, dropout)
        
        # Attention LSTM
        self.lstm = AttentionLSTM(
            input_dim=cnn_base_filters,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout
        )
        
        lstm_output_dim = lstm_hidden * 2
        
        # Output layers with uncertainty estimation
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, output_dim * 3)  # For mean, lower, upper bounds
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation
        """
        # Transpose for CNN [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)
        
        # Multi-scale CNN
        cnn_features = self.cnn(x_cnn)
        cnn_features = self.dropout(cnn_features)
        
        # Transpose back for LSTM [batch, seq_len, features]
        cnn_features = cnn_features.transpose(1, 2)
        
        # LSTM with attention
        if return_attention:
            lstm_output, attention_weights = self.lstm(cnn_features, return_attention=True)
        else:
            lstm_output = self.lstm(cnn_features, return_attention=False)
            attention_weights = None
        
        # Output with uncertainty
        output = self.output_layer(lstm_output)
        output = output.view(output.size(0), -1, 3)
        
        # Split into mean, lower, upper bounds
        mean = output[:, :, 0]
        lower = output[:, :, 1]
        upper = output[:, :, 2]
        
        result = {
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'uncertainty': (upper - lower) / 2.0
        }
        
        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights
        
        return result

class ResidualCNNLSTM(nn.Module):
    """CNN-LSTM with residual connections"""
    
    def __init__(self, input_dim: int, cnn_blocks: int, cnn_channels: int,
                 lstm_hidden: int, lstm_layers: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Initial CNN layer
        self.initial_cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual CNN blocks
        self.residual_blocks = nn.ModuleList([
            ResidualCNNBlock(cnn_channels, dropout=dropout)
            for _ in range(cnn_blocks)
        ])
        
        # Final CNN layer
        self.final_cnn = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels // 2, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(cnn_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels // 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = lstm_hidden * 2
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose for CNN
        x = x.transpose(1, 2)
        
        # Initial CNN
        x = self.initial_cnn(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final CNN
        x = self.final_cnn(x)
        x = self.dropout(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep
        
        # Output
        output = self.output_layer(lstm_out)
        
        return output

class HierarchicalCNNLSTM(nn.Module):
    """Hierarchical CNN-LSTM for multi-resolution analysis"""
    
    def __init__(self, input_dim: int, cnn_levels: List[int], lstm_hidden: int,
                 lstm_layers: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.cnn_levels = cnn_levels
        
        # Create CNN for each level
        self.cnn_level_modules = nn.ModuleList()
        for level, filters in enumerate(cnn_levels):
            if level == 0:
                cnn_input = input_dim
            else:
                cnn_input = cnn_levels[level - 1]
            
            cnn_module = TemporalCNN(
                input_dim=cnn_input,
                filters=[filters],
                kernel_sizes=[3],
                pool_sizes=[2],
                dropout=dropout
            )
            self.cnn_level_modules.append(cnn_module)
        
        # LSTM for each level
        self.lstm_level_modules = nn.ModuleList()
        for filters in cnn_levels:
            lstm_module = nn.LSTM(
                input_size=filters,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=True
            )
            self.lstm_level_modules.append(lstm_module)
        
        # Fusion layer
        total_features = lstm_hidden * 2 * len(cnn_levels)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        level_outputs = []
        
        # Process each level
        for level, (cnn_module, lstm_module) in enumerate(zip(self.cnn_level_modules, self.lstm_level_modules)):
            if level == 0:
                level_input = x
            else:
                # Downsample input for higher levels
                level_input = F.avg_pool1d(x.transpose(1, 2), kernel_size=2**level).transpose(1, 2)
            
            # CNN
            cnn_out = cnn_module(level_input)
            cnn_out = self.dropout(cnn_out)
            
            # LSTM
            lstm_out, _ = lstm_module(cnn_out)
            lstm_out = lstm_out[:, -1, :]  # Last timestep
            
            level_outputs.append(lstm_out)
        
        # Concatenate all level outputs
        combined = torch.cat(level_outputs, dim=1)
        
        # Fusion
        output = self.fusion_layer(combined)
        
        return output

# ============ PyTorch Lightning Modules ============
class CNNLSTMLightning(pl.LightningModule):
    """PyTorch Lightning module for CNN-LSTM models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Extract configuration
        self.input_dim = config['input_dim']
        self.cnn_filters = config.get('cnn_filters', [64, 128, 256])
        self.cnn_kernel_sizes = config.get('cnn_kernel_sizes', [3, 5, 3])
        self.lstm_hidden = config.get('lstm_hidden', 128)
        self.lstm_layers = config.get('lstm_layers', 2)
        self.output_dim = config.get('output_dim', 1)
        self.dropout = config.get('dropout', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        
        # Model architecture
        model_type = config.get('model_type', 'standard')
        
        if model_type == 'standard':
            self.model = CNNLSTMModel(
                input_dim=self.input_dim,
                cnn_filters=self.cnn_filters,
                cnn_kernel_sizes=self.cnn_kernel_sizes,
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        elif model_type == 'multiscale':
            self.model = MultiScaleCNNLSTM(
                input_dim=self.input_dim,
                cnn_base_filters=self.cnn_filters[0],
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        elif model_type == 'residual':
            self.model = ResidualCNNLSTM(
                input_dim=self.input_dim,
                cnn_blocks=len(self.cnn_filters),
                cnn_channels=self.cnn_filters[0],
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        elif model_type == 'hierarchical':
            self.model = HierarchicalCNNLSTM(
                input_dim=self.input_dim,
                cnn_levels=self.cnn_filters,
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()
        
        # Metrics
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        
        # Store model type
        self.model_type = model_type
    
    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.model_type == 'multiscale':
            output = self(x)
            y_hat = output['mean']
        else:
            y_hat = self(x)
        
        # Combined loss
        mse_loss = self.mse_loss(y_hat, y)
        mae_loss = self.mae_loss(y_hat, y)
        huber_loss = self.huber_loss(y_hat, y)
        
        # Weighted loss
        loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber_loss
        
        # For multiscale model, add uncertainty regularization
        if self.model_type == 'multiscale':
            uncertainty_loss = output['uncertainty'].mean()
            loss = loss + 0.1 * uncertainty_loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', mse_loss)
        self.log('train_mae', mae_loss)
        self.log('train_huber', huber_loss)
        
        # Calculate R² score
        ss_res = torch.sum((y - y_hat) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log('train_r2', r2)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if self.model_type == 'multiscale':
            output = self(x)
            y_hat = output['mean']
        else:
            y_hat = self(x)
        
        # Combined loss
        mse_loss = self.mse_loss(y_hat, y)
        mae_loss = self.mae_loss(y_hat, y)
        huber_loss = self.huber_loss(y_hat, y)
        loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber_loss
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', mse_loss)
        self.log('val_mae', mae_loss)
        self.log('val_huber', huber_loss)
        
        # Calculate R² score
        ss_res = torch.sum((y - y_hat) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log('val_r2', r2)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        if self.model_type == 'multiscale':
            output = self(x)
            y_hat = output['mean']
            # Log uncertainty metrics
            self.log('test_uncertainty', output['uncertainty'].mean())
        else:
            y_hat = self(x)
        
        # Combined loss
        mse_loss = self.mse_loss(y_hat, y)
        mae_loss = self.mae_loss(y_hat, y)
        huber_loss = self.huber_loss(y_hat, y)
        loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber_loss
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_mse', mse_loss)
        self.log('test_mae', mae_loss)
        self.log('test_huber', huber_loss)
        
        # Calculate R² score
        ss_res = torch.sum((y - y_hat) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log('test_r2', r2)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights if model supports it"""
        if hasattr(self.model, 'use_attention') and self.model.use_attention:
            with torch.no_grad():
                _, attention_weights = self.model(x, return_attention=True)
            return attention_weights
        return None
    
    def visualize_cnn_features(self, x: torch.Tensor, layer_idx: int = 0):
        """Visualize CNN feature maps"""
        if not hasattr(self.model, 'cnn'):
            print("Model doesn't have CNN layer")
            return
        
        # Hook to get feature maps
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output.detach())
        
        # Register hook
        if isinstance(self.model.cnn, nn.Sequential):
            if layer_idx < len(self.model.cnn):
                hook = self.model.cnn[layer_idx].register_forward_hook(hook_fn)
            else:
                print(f"Layer {layer_idx} not found")
                return
        else:
            hook = self.model.cnn.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(x)
        
        # Remove hook
        hook.remove()
        
        if feature_maps:
            # Visualize first few feature maps
            fm = feature_maps[0][0].cpu().numpy()  # First batch
            n_features = min(16, fm.shape[0])
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 8))
            for i, ax in enumerate(axes.flat):
                if i < n_features:
                    ax.plot(fm[i])
                    ax.set_title(f'Feature {i+1}')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
            
            plt.suptitle(f'CNN Feature Maps - Layer {layer_idx}')
            plt.tight_layout()
            plt.show()

# ============ Dataset and DataLoader ============
class TimeSeriesCNNLSTMDataset(Dataset):
    """Dataset for CNN-LSTM time series forecasting"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 60, stride: int = 1,
                 target_offset: int = 1):
        """
        Args:
            data: Input features (n_samples, n_features)
            targets: Target values (n_samples, n_targets)
            sequence_length: Length of input sequences
            stride: Step size between sequences
            target_offset: Offset for target prediction
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_offset = target_offset
        
        # Precompute indices
        self.indices = []
        for i in range(0, len(data) - sequence_length - target_offset + 1, stride):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i:i + self.sequence_length]
        y = self.targets[i + self.sequence_length + self.target_offset - 1]
        return x, y

class MultiStepCNNLSTMDataset(Dataset):
    """Dataset for multi-step forecasting with CNN-LSTM"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray,
                 sequence_length: int = 60, forecast_horizon: int = 24,
                 stride: int = 1):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        # Precompute indices
        self.indices = []
        for i in range(0, len(data) - sequence_length - forecast_horizon + 1, stride):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i:i + self.sequence_length]
        y = self.targets[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
        return x, y

# ============ Model Factory ============
class CNNLSTMModelFactory:
    """Factory class for creating CNN-LSTM models"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create CNN-LSTM model based on type"""
        
        base_config = {
            'input_dim': config.get('input_dim', 50),
            'cnn_filters': config.get('cnn_filters', [64, 128, 256]),
            'cnn_kernel_sizes': config.get('cnn_kernel_sizes', [3, 5, 3]),
            'lstm_hidden': config.get('lstm_hidden', 128),
            'lstm_layers': config.get('lstm_layers', 2),
            'output_dim': config.get('output_dim', 1),
            'dropout': config.get('dropout', 0.3)
        }
        
        if model_type == 'standard':
            return CNNLSTMModel(**base_config)
        
        elif model_type == 'multiscale':
            base_config['cnn_base_filters'] = base_config['cnn_filters'][0]
            return MultiScaleCNNLSTM(**base_config)
        
        elif model_type == 'residual':
            base_config['cnn_blocks'] = len(base_config['cnn_filters'])
            base_config['cnn_channels'] = base_config['cnn_filters'][0]
            return ResidualCNNLSTM(**base_config)
        
        elif model_type == 'hierarchical':
            return HierarchicalCNNLSTM(**base_config)
        
        elif model_type == 'lightning':
            config['model_type'] = 'standard'
            return CNNLSTMLightning(config)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type"""
        
        defaults = {
            'standard': {
                'input_dim': 50,
                'cnn_filters': [64, 128, 256],
                'cnn_kernel_sizes': [3, 5, 3],
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'output_dim': 1,
                'dropout': 0.3,
                'learning_rate': 0.001
            },
            'multiscale': {
                'input_dim': 50,
                'cnn_filters': [64, 128, 256],
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'output_dim': 1,
                'dropout': 0.3,
                'learning_rate': 0.001
            },
            'residual': {
                'input_dim': 50,
                'cnn_filters': [64, 128, 256],
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'output_dim': 1,
                'dropout': 0.3,
                'learning_rate': 0.001
            },
            'hierarchical': {
                'input_dim': 50,
                'cnn_levels': [64, 128, 256],
                'lstm_hidden': 64,
                'lstm_layers': 2,
                'output_dim': 1,
                'dropout': 0.3,
                'learning_rate': 0.001
            }
        }
        
        return defaults.get(model_type, defaults['standard'])

# ============ Utility Functions ============
def prepare_cnn_lstm_data(data: np.ndarray, sequence_length: int, 
                          target_col: int = -1, test_size: float = 0.2) -> Tuple:
    """Prepare data for CNN-LSTM training"""
    
    # Separate features and target
    if target_col == -1:
        X = data[:, :-1]
        y = data[:, -1:]
    else:
        X = np.delete(data, target_col, axis=1)
        y = data[:, target_col:target_col+1]
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Split into train/test
    split_idx = int(len(X_seq) * (1 - test_size))
    
    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]
    
    return X_train, y_train, X_test, y_test

def calculate_cnn_lstm_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics for CNN-LSTM predictions"""
    
    metrics = {}
    
    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    # Mean Squared Error
    metrics['mse'] = np.mean((predictions - targets) ** 2)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Percentage Error
    epsilon = 1e-8
    metrics['mape'] = np.mean(np.abs((predictions - targets) / (targets + epsilon))) * 100
    
    # R-squared
    ss_res = np.sum((predictions - targets) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + epsilon))
    
    # Directional Accuracy
    if len(predictions) > 1:
        pred_direction = np.sign(predictions[1:] - predictions[:-1])
        true_direction = np.sign(targets[1:] - targets[:-1])
        metrics['direction_accuracy'] = np.mean(pred_direction == true_direction) * 100
    
    return metrics

# ============ Example Usage ============
def example_usage():
    """Example usage of CNN-LSTM models"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    sequence_length = 60
    
    # Create synthetic time series data
    data = np.random.randn(n_samples, n_features).cumsum(axis=0)
    targets = np.random.randn(n_samples, 1).cumsum(axis=0)
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_cnn_lstm_data(
        np.hstack([data, targets]),
        sequence_length=sequence_length,
        test_size=0.2
    )
    
    # Create datasets
    train_dataset = TimeSeriesCNNLSTMDataset(X_train, y_train, sequence_length)
    test_dataset = TimeSeriesCNNLSTMDataset(X_test, y_test, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test different model architectures
    model_types = ['standard', 'multiscale', 'residual', 'hierarchical']
    
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type}:")
        print("-" * 30)
        
        # Get default config
        config = CNNLSTMModelFactory.get_default_config(model_type)
        config['input_dim'] = n_features
        config['output_dim'] = 1
        
        # Create model
        model = CNNLSTMModelFactory.create_model(model_type, config)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")
        
        # Test forward pass
        test_batch = next(iter(train_loader))[0]
        output = model(test_batch)
        
        if isinstance(output, dict):
            print(f"Output type: dict with keys: {list(output.keys())}")
            for key, value in output.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape}")
        else:
            print(f"Output shape: {output.shape}")
        
        # Test attention weights if available
        if hasattr(model, 'use_attention') and model.use_attention:
            try:
                output, attention_weights = model(test_batch, return_attention=True)
                print(f"Attention weights shape: {attention_weights.shape}")
            except Exception as e:
                print(f"Could not get attention weights: {str(e)}")
        
        results[model_type] = {
            'model': model,
            'num_params': num_params
        }
    
    return results

if __name__ == "__main__":
    print("CNN-LSTM Hybrid Models for Time Series Forecasting")
    print("=" * 60)
    
    # Test example usage
    results = example_usage()
    
    print("\n" + "=" * 60)
    print("CNN-LSTM models are ready for time series forecasting!")
    
    # Create a lightning model example
    print("\nCreating PyTorch Lightning model...")
    config = {
        'input_dim': 50,
        'cnn_filters': [64, 128, 256],
        'cnn_kernel_sizes': [3, 5, 3],
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'output_dim': 1,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'model_type': 'standard'
    }
    
    lightning_model = CNNLSTMLightning(config)
    print(f"Lightning model created with {sum(p.numel() for p in lightning_model.parameters()):,} parameters")
    
    print("\nAll models are ready for training!")