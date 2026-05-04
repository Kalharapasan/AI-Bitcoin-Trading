"""
LSTM with Attention Mechanisms for Time Series Forecasting
Advanced LSTM architectures with various attention mechanisms for financial data
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

# ============ Attention Mechanisms ============
class BahdanauAttention(nn.Module):
    """Bahdanau Attention (Additive Attention)"""
    
    def __init__(self, hidden_dim: int, attention_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if attention_dim is None:
            attention_dim = hidden_dim
        
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim) - Current decoder hidden state
            encoder_outputs: (batch_size, seq_len, hidden_dim) - All encoder outputs
        
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        # Expand decoder hidden state for attention computation
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate with encoder outputs
        concat = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=-1)
        
        # Compute attention energies
        attention_energies = self.attention_network(concat).squeeze(-1)
        
        # Compute attention weights
        attention_weights = self.softmax(attention_energies)
        attention_weights = self.dropout(attention_weights)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attention_weights

class LuongAttention(nn.Module):
    """Luong Attention (Multiplicative Attention)"""
    
    def __init__(self, hidden_dim: int, method: str = "general", dropout: float = 0.1):
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        
        if method == "general":
            self.attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == "concat":
            self.attention = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(hidden_dim))
        elif method == "dot":
            pass  # No parameters needed for dot product
        else:
            raise ValueError(f"Unknown attention method: {method}")
        
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        
        if method == "concat":
            nn.init.uniform_(self.v, -0.1, 0.1)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        
        if self.method == "dot":
            # Dot product attention
            decoder_hidden = decoder_hidden.unsqueeze(2)  # (batch_size, hidden_dim, 1)
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden).squeeze(2)
        
        elif self.method == "general":
            # General attention
            decoder_hidden = self.attention(decoder_hidden).unsqueeze(2)  # (batch_size, hidden_dim, 1)
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden).squeeze(2)
        
        elif self.method == "concat":
            # Concatenation attention
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
            concat = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=-1)
            attention_scores = self.attention(concat)  # (batch_size, seq_len, hidden_dim)
            attention_scores = torch.tanh(attention_scores)
            attention_scores = torch.matmul(attention_scores, self.v)  # (batch_size, seq_len)
        
        # Compute attention weights
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attention_weights

class MultiHeadAttentionLSTM(nn.Module):
    """Multi-Head Attention for LSTM outputs"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch_size, seq_len, hidden_dim)
        
        Returns:
            attended_outputs: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = lstm_outputs.shape
        
        # Linear projections
        Q = self.query_proj(lstm_outputs)  # (batch_size, seq_len, hidden_dim)
        K = self.key_proj(lstm_outputs)    # (batch_size, seq_len, hidden_dim)
        V = self.value_proj(lstm_outputs)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Final projection
        output = self.output_proj(attended_values)
        
        return output, attention_weights

class TemporalAttention(nn.Module):
    """Temporal Attention for capturing time dependencies"""
    
    def __init__(self, hidden_dim: int, attention_dim: int = None, 
                 use_temporal_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        if attention_dim is None:
            attention_dim = hidden_dim
        
        # Attention network
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1)
        )
        
        # Temporal bias (learnable position bias)
        self.use_temporal_bias = use_temporal_bias
        if use_temporal_bias:
            self.temporal_bias = nn.Parameter(torch.zeros(1, 1, 1))
        
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_dim = lstm_outputs.shape
        
        # Compute attention scores
        attention_scores = self.attention_network(lstm_outputs).squeeze(-1)  # (batch_size, seq_len)
        
        # Add temporal bias if enabled
        if self.use_temporal_bias:
            # Create position weights (recent positions get higher bias)
            position_weights = torch.linspace(0, 1, seq_len, device=lstm_outputs.device)
            position_weights = position_weights.unsqueeze(0).expand(batch_size, -1)
            attention_scores = attention_scores + self.temporal_bias * position_weights
        
        # Compute attention weights
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        
        return context_vector, attention_weights

class HierarchicalAttention(nn.Module):
    """Hierarchical Attention for multi-level feature attention"""
    
    def __init__(self, hidden_dim: int, num_levels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        # Attention networks for each level
        self.level_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_levels)
        ])
        
        # Level combination weights
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            lstm_outputs: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context_vector: (batch_size, hidden_dim)
            all_attention_weights: List of (batch_size, seq_len) for each level
        """
        batch_size, seq_len, _ = lstm_outputs.shape
        
        all_attention_weights = []
        level_contexts = []
        
        # Compute attention for each level
        for i in range(self.num_levels):
            # Different window sizes for different levels
            if i == 0:
                # Level 1: Short-term attention
                window_size = max(seq_len // 4, 1)
            elif i == 1:
                # Level 2: Medium-term attention
                window_size = max(seq_len // 2, 1)
            else:
                # Level 3: Long-term attention
                window_size = seq_len
            
            # Apply attention
            attention_scores = self.level_attentions[i](lstm_outputs).squeeze(-1)
            attention_weights = self.softmax(attention_scores)
            attention_weights = self.dropout(attention_weights)
            
            # Store attention weights
            all_attention_weights.append(attention_weights)
            
            # Compute level context
            level_context = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
            level_contexts.append(level_context)
        
        # Combine level contexts
        level_contexts = torch.stack(level_contexts, dim=1)  # (batch_size, num_levels, hidden_dim)
        level_weights = F.softmax(self.level_weights, dim=0)  # (num_levels)
        
        # Weighted combination
        context_vector = torch.sum(
            level_contexts * level_weights.view(1, -1, 1),
            dim=1
        )
        
        return context_vector, all_attention_weights

# ============ LSTM Architectures ============
class LSTMAttentionModel(nn.Module):
    """LSTM with Attention Mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.3, bidirectional: bool = True,
                 attention_type: str = "bahdanau",  # bahdanau, luong, multihead, temporal, hierarchical
                 attention_dim: int = None, num_heads: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(lstm_output_dim, attention_dim, dropout)
        elif attention_type == "luong":
            self.attention = LuongAttention(lstm_output_dim, method="general", dropout=dropout)
        elif attention_type == "multihead":
            self.attention = MultiHeadAttentionLSTM(lstm_output_dim, num_heads, dropout)
        elif attention_type == "temporal":
            self.attention = TemporalAttention(lstm_output_dim, attention_dim, dropout=dropout)
        elif attention_type == "hierarchical":
            self.attention = HierarchicalAttention(lstm_output_dim, num_levels=3, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, output_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM and linear layer weights"""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Initialize linear layer weights
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Predictions
            attention_weights: Attention weights (if return_attention=True)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Get the last hidden state (for attention)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden_forward = hidden[-2, :, :]  # Last layer forward
            hidden_backward = hidden[-1, :, :]  # Last layer backward
            decoder_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            decoder_hidden = hidden[-1, :, :]  # Last layer hidden state
        
        # Apply attention
        if self.attention_type in ["multihead", "temporal", "hierarchical"]:
            # These attention types work directly on lstm_out
            attended_output, attention_weights = self.attention(lstm_out)
            
            # For multihead, we need to pool across sequence dimension
            if self.attention_type == "multihead":
                context_vector = attended_output.mean(dim=1)
            else:
                context_vector = attended_output
        else:
            # Bahdanau/Luong attention
            context_vector, attention_weights = self.attention(decoder_hidden, lstm_out)
        
        # Final output
        output = self.output_layers(context_vector)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions without returning attention weights"""
        return self.forward(x, return_attention=False)

class StackedLSTMWithAttention(nn.Module):
    """Stacked LSTM with residual connections and attention"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, dropout: float = 0.3,
                 attention_type: str = "temporal"):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        
        # Create LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # LSTM layer
            lstm_layer = nn.LSTM(
                input_size=prev_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0
            )
            self.lstm_layers.append(lstm_layer)
            
            # Attention layer for this level
            attention_layer = TemporalAttention(hidden_dim, dropout=dropout)
            self.attention_layers.append(attention_layer)
            
            # Residual connection (if dimension changes)
            if prev_dim != hidden_dim:
                residual_layer = nn.Linear(prev_dim, hidden_dim)
            else:
                residual_layer = nn.Identity()
            self.residual_layers.append(residual_layer)
            
            prev_dim = hidden_dim
        
        # Final output dimension
        final_hidden_dim = hidden_dims[-1]
        
        # Final attention layer
        self.final_attention = HierarchicalAttention(final_hidden_dim, num_levels=3, dropout=dropout)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights"""
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
        
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Forward pass through stacked LSTM with attention
        
        Returns:
            output: Predictions
            all_attention_weights: List of attention weights from each layer
        """
        all_attention_weights = []
        
        # Process through each LSTM layer
        for i, (lstm_layer, attention_layer, residual_layer) in enumerate(
            zip(self.lstm_layers, self.attention_layers, self.residual_layers)
        ):
            # LSTM forward pass
            lstm_out, (hidden, cell) = lstm_layer(x)
            lstm_out = self.dropout(lstm_out)
            
            # Apply attention
            context_vector, attention_weights = attention_layer(lstm_out)
            
            if return_attention:
                all_attention_weights.append(attention_weights)
            
            # Residual connection
            residual = residual_layer(x[:, -1, :]) if x.shape[1] > 0 else residual_layer(x.mean(dim=1))
            
            # Prepare for next layer (use context vector as input)
            x = lstm_out  # Use LSTM outputs for next layer
        
        # Apply final hierarchical attention
        final_context, final_attention_weights = self.final_attention(lstm_out)
        
        if return_attention:
            all_attention_weights.append(final_attention_weights)
        
        # Final output
        output = self.output_layers(final_context)
        
        if return_attention:
            return output, all_attention_weights
        return output

class ConvLSTMWithAttention(nn.Module):
    """CNN-LSTM hybrid model with attention"""
    
    def __init__(self, input_dim: int, cnn_channels: List[int],
                 lstm_hidden: int, lstm_layers: int, output_dim: int,
                 kernel_sizes: List[int] = None, dropout: float = 0.3,
                 attention_type: str = "temporal"):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 3]
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, kernel_sizes)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2, padding_mode='replicate'),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.MaxPool1d(2) if i < len(cnn_channels) - 1 else nn.Identity(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_output_size = cnn_channels[-1]
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = lstm_hidden * 2
        
        # Attention mechanism
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(lstm_output_dim, dropout=dropout)
        elif attention_type == "luong":
            self.attention = LuongAttention(lstm_output_dim, dropout=dropout)
        elif attention_type == "multihead":
            self.attention = MultiHeadAttentionLSTM(lstm_output_dim, num_heads=8, dropout=dropout)
        elif attention_type == "temporal":
            self.attention = TemporalAttention(lstm_output_dim, dropout=dropout)
        else:
            self.attention = HierarchicalAttention(lstm_output_dim, num_levels=3, dropout=dropout)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize CNN weights
        for layer in self.cnn:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Initialize output layer weights
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Forward pass through CNN-LSTM with attention
        
        Returns:
            output: Predictions
            attention_weights: Attention weights (if return_attention=True)
        """
        # CNN expects [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)
        
        # CNN feature extraction
        cnn_features = self.cnn(x_cnn)
        
        # Transpose back for LSTM [batch, seq_len, features]
        # Note: CNN reduces sequence length due to pooling
        cnn_features = cnn_features.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        lstm_out = self.dropout(lstm_out)
        
        # Get hidden state for attention
        if self.lstm.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            decoder_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            decoder_hidden = hidden[-1, :, :]
        
        # Apply attention
        if isinstance(self.attention, (MultiHeadAttentionLSTM, TemporalAttention, HierarchicalAttention)):
            context_vector, attention_weights = self.attention(lstm_out)
        else:
            context_vector, attention_weights = self.attention(decoder_hidden, lstm_out)
        
        # Output
        output = self.output_layer(context_vector)
        
        if return_attention:
            return output, attention_weights
        return output

class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with multi-level attention"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Forward LSTM
        self.forward_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Backward LSTM
        self.backward_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanisms
        self.forward_attention = TemporalAttention(hidden_dim, dropout=dropout)
        self.backward_attention = TemporalAttention(hidden_dim, dropout=dropout)
        self.combined_attention = HierarchicalAttention(hidden_dim * 2, num_levels=3, dropout=dropout)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights"""
        for lstm in [self.forward_lstm, self.backward_lstm]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
        
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Forward pass through bidirectional LSTM with attention
        
        Returns:
            output: Predictions
            attention_dict: Dictionary of attention weights (if return_attention=True)
        """
        # Forward LSTM
        forward_out, (forward_hidden, forward_cell) = self.forward_lstm(x)
        forward_out = self.dropout(forward_out)
        
        # Backward LSTM (reverse sequence)
        backward_out, (backward_hidden, backward_cell) = self.backward_lstm(
            torch.flip(x, dims=[1])
        )
        backward_out = torch.flip(backward_out, dims=[1])
        backward_out = self.dropout(backward_out)
        
        # Individual attention
        forward_context, forward_weights = self.forward_attention(forward_out)
        backward_context, backward_weights = self.backward_attention(backward_out)
        
        # Combine forward and backward
        combined = torch.cat([forward_out, backward_out], dim=-1)
        
        # Combined attention
        combined_context, combined_weights = self.combined_attention(combined)
        
        # Output
        output = self.output_layer(combined_context)
        
        if return_attention:
            attention_dict = {
                'forward': forward_weights,
                'backward': backward_weights,
                'combined': combined_weights
            }
            return output, attention_dict
        
        return output

# ============ PyTorch Lightning Modules ============
class LSTMAttentionLightning(pl.LightningModule):
    """PyTorch Lightning module for LSTM with attention"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Extract configuration
        self.input_dim = config['input_dim']
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.output_dim = config.get('output_dim', 1)
        self.dropout = config.get('dropout', 0.3)
        self.bidirectional = config.get('bidirectional', True)
        self.attention_type = config.get('attention_type', 'temporal')
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        
        # Model architecture
        model_arch = config.get('model_arch', 'lstm_attention')
        
        if model_arch == 'lstm_attention':
            self.model = LSTMAttentionModel(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=self.output_dim,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                attention_type=self.attention_type
            )
        elif model_arch == 'stacked_lstm':
            hidden_dims = config.get('hidden_dims', [128, 64, 32])
            self.model = StackedLSTMWithAttention(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                output_dim=self.output_dim,
                dropout=self.dropout,
                attention_type=self.attention_type
            )
        elif model_arch == 'cnn_lstm':
            cnn_channels = config.get('cnn_channels', [64, 128, 256])
            self.model = ConvLSTMWithAttention(
                input_dim=self.input_dim,
                cnn_channels=cnn_channels,
                lstm_hidden=self.hidden_dim,
                lstm_layers=self.num_layers,
                output_dim=self.output_dim,
                dropout=self.dropout,
                attention_type=self.attention_type
            )
        elif model_arch == 'bi_lstm':
            self.model = BiLSTMAttention(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model architecture: {model_arch}")
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()
        
        # Metrics
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
    
    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Combined loss
        mse_loss = self.mse_loss(y_hat, y)
        mae_loss = self.mae_loss(y_hat, y)
        huber_loss = self.huber_loss(y_hat, y)
        
        # Weighted loss
        loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber_loss
        
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
    
    def get_attention_weights(self, x: torch.Tensor) -> Any:
        """Get attention weights for interpretability"""
        with torch.no_grad():
            _, attention_weights = self.model(x, return_attention=True)
        return attention_weights
    
    def visualize_attention(self, x: torch.Tensor, sample_idx: int = 0):
        """Visualize attention weights for a sample"""
        attention_weights = self.get_attention_weights(x)
        
        if attention_weights is None:
            print("No attention weights available")
            return
        
        # Handle different attention weight formats
        if isinstance(attention_weights, dict):
            # BiLSTM attention
            for name, weights in attention_weights.items():
                if isinstance(weights, list):
                    # Hierarchical attention
                    for level, level_weights in enumerate(weights):
                        if len(level_weights.shape) >= 2:
                            self._plot_attention(
                                level_weights[sample_idx].cpu().numpy(),
                                title=f"{name} Attention - Level {level + 1}"
                            )
                elif len(weights.shape) >= 2:
                    self._plot_attention(
                        weights[sample_idx].cpu().numpy(),
                        title=f"{name} Attention"
                    )
        
        elif isinstance(attention_weights, list):
            # Stacked LSTM attention
            for layer_idx, layer_weights in enumerate(attention_weights):
                if len(layer_weights.shape) >= 2:
                    self._plot_attention(
                        layer_weights[sample_idx].cpu().numpy(),
                        title=f"Layer {layer_idx + 1} Attention"
                    )
        
        elif len(attention_weights.shape) >= 2:
            # Single attention matrix
            self._plot_attention(
                attention_weights[sample_idx].cpu().numpy(),
                title="Attention Weights"
            )
    
    def _plot_attention(self, weights: np.ndarray, title: str = "Attention"):
        """Plot attention weights"""
        if len(weights.shape) == 1:
            # 1D attention weights (temporal attention)
            plt.figure(figsize=(10, 4))
            plt.plot(weights, marker='o', linestyle='-', color='b', alpha=0.7)
            plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        elif len(weights.shape) == 2:
            # 2D attention weights (multi-head attention)
            plt.figure(figsize=(10, 8))
            plt.imshow(weights, cmap='viridis', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.title(title)
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.tight_layout()
            plt.show()

# ============ Model Factory ============
class LSTMModelFactory:
    """Factory class for creating LSTM models with attention"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create LSTM model based on type"""
        
        base_config = {
            'input_dim': config.get('input_dim', 50),
            'hidden_dim': config.get('hidden_dim', 128),
            'num_layers': config.get('num_layers', 2),
            'output_dim': config.get('output_dim', 1),
            'dropout': config.get('dropout', 0.3),
            'bidirectional': config.get('bidirectional', True),
            'attention_type': config.get('attention_type', 'temporal')
        }
        
        if model_type == 'lstm_attention':
            return LSTMAttentionModel(**base_config)
        
        elif model_type == 'stacked_lstm':
            base_config['hidden_dims'] = config.get('hidden_dims', [128, 64, 32])
            return StackedLSTMWithAttention(**base_config)
        
        elif model_type == 'cnn_lstm':
            base_config['cnn_channels'] = config.get('cnn_channels', [64, 128, 256])
            base_config['lstm_hidden'] = base_config.pop('hidden_dim')
            base_config['lstm_layers'] = base_config.pop('num_layers')
            return ConvLSTMWithAttention(**base_config)
        
        elif model_type == 'bi_lstm':
            return BiLSTMAttention(**base_config)
        
        elif model_type == 'lightning':
            return LSTMAttentionLightning(config)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type"""
        
        defaults = {
            'lstm_attention': {
                'input_dim': 50,
                'hidden_dim': 128,
                'num_layers': 2,
                'output_dim': 1,
                'dropout': 0.3,
                'bidirectional': True,
                'attention_type': 'temporal',
                'learning_rate': 0.001
            },
            'stacked_lstm': {
                'input_dim': 50,
                'hidden_dims': [128, 64, 32],
                'output_dim': 1,
                'dropout': 0.3,
                'attention_type': 'hierarchical',
                'learning_rate': 0.001
            },
            'cnn_lstm': {
                'input_dim': 50,
                'cnn_channels': [64, 128, 256],
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'output_dim': 1,
                'dropout': 0.3,
                'attention_type': 'temporal',
                'learning_rate': 0.001
            },
            'bi_lstm': {
                'input_dim': 50,
                'hidden_dim': 128,
                'num_layers': 2,
                'output_dim': 1,
                'dropout': 0.3,
                'learning_rate': 0.001
            }
        }
        
        return defaults.get(model_type, defaults['lstm_attention'])

# ============ Dataset and DataLoader ============
class TimeSeriesLSTMDataset(Dataset):
    """Dataset for LSTM time series forecasting"""
    
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

class MultiStepLSTMDataset(Dataset):
    """Dataset for multi-step forecasting with LSTM"""
    
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

# ============ Utility Functions ============
def prepare_lstm_data(data: np.ndarray, sequence_length: int, 
                      target_col: int = -1, test_size: float = 0.2) -> Tuple:
    """Prepare data for LSTM training"""
    
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

def calculate_lstm_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics for LSTM predictions"""
    
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
    pred_direction = np.sign(predictions[1:] - predictions[:-1])
    true_direction = np.sign(targets[1:] - targets[:-1])
    metrics['direction_accuracy'] = np.mean(pred_direction == true_direction) * 100
    
    return metrics

# ============ Training Functions ============
def train_lstm_model(model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, num_epochs: int = 100,
                    learning_rate: float = 0.001, device: str = 'cuda'):
    """Train LSTM model"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

# ============ Example Usage ============
def example_usage():
    """Example usage of LSTM with attention models"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    sequence_length = 60
    
    # Create synthetic time series data
    data = np.random.randn(n_samples, n_features).cumsum(axis=0)
    targets = np.random.randn(n_samples, 1).cumsum(axis=0)
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_lstm_data(
        np.hstack([data, targets]),
        sequence_length=sequence_length,
        test_size=0.2
    )
    
    # Create datasets
    train_dataset = TimeSeriesLSTMDataset(X_train, y_train, sequence_length)
    test_dataset = TimeSeriesLSTMDataset(X_test, y_test, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test different model architectures
    model_types = ['lstm_attention', 'stacked_lstm', 'cnn_lstm', 'bi_lstm']
    
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type}:")
        print("-" * 30)
        
        # Get default config
        config = LSTMModelFactory.get_default_config(model_type)
        config['input_dim'] = n_features
        config['output_dim'] = 1
        
        # Create model
        model = LSTMModelFactory.create_model(model_type, config)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")
        
        # Test forward pass
        test_batch = next(iter(train_loader))[0]
        output = model(test_batch)
        print(f"Output shape: {output.shape}")
        
        # Test attention weights
        if hasattr(model, 'forward') and callable(getattr(model, 'forward', None)):
            try:
                output, attention_weights = model(test_batch, return_attention=True)
                print(f"Attention weights type: {type(attention_weights)}")
                
                if isinstance(attention_weights, dict):
                    print(f"Attention keys: {list(attention_weights.keys())}")
                elif isinstance(attention_weights, list):
                    print(f"Number of attention layers: {len(attention_weights)}")
                
            except Exception as e:
                print(f"Could not get attention weights: {str(e)}")
        
        results[model_type] = {
            'model': model,
            'num_params': num_params,
            'output_shape': output.shape
        }
    
    return results

if __name__ == "__main__":
    print("LSTM with Attention Models for Time Series Forecasting")
    print("=" * 60)
    
    # Test example usage
    results = example_usage()
    
    print("\n" + "=" * 60)
    print("LSTM with attention models are ready for time series forecasting!")
    
    # Create a lightning model example
    print("\nCreating PyTorch Lightning model...")
    config = {
        'input_dim': 50,
        'hidden_dim': 128,
        'num_layers': 2,
        'output_dim': 1,
        'dropout': 0.3,
        'bidirectional': True,
        'attention_type': 'temporal',
        'learning_rate': 0.001,
        'model_arch': 'lstm_attention'
    }
    
    lightning_model = LSTMAttentionLightning(config)
    print(f"Lightning model created with {sum(p.numel() for p in lightning_model.parameters()):,} parameters")
    
    print("\nAll models are ready for training!")