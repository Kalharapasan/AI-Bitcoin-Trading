"""
Transformer Models for Time Series Forecasting
Advanced transformer architectures optimized for financial time series data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# ============ Positional Encoding ============
class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer with multiple frequency bands"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1,
                 learnable: bool = False, n_frequencies: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable
        self.d_model = d_model
        
        if learnable:
            # Learnable positional encoding
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.xavier_uniform_(self.pe)
        else:
            # Fixed sinusoidal encoding with multiple frequency bands
            position = torch.arange(max_len).unsqueeze(1)
            
            # Multiple frequency bands for better time representation
            div_terms = []
            frequencies = torch.linspace(1, n_frequencies, n_frequencies)
            
            for freq in frequencies:
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * 
                    (-math.log(10000.0) / d_model) * freq
                )
                div_terms.append(div_term)
            
            # Combine frequency bands
            div_term = torch.stack(div_terms).mean(0)
            
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Positionally encoded tensor
        """
        if self.learnable:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """Fully learnable positional encoding with temporal attention"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings
        pos_emb = self.position_embeddings(positions)
        
        # Apply temporal attention to capture position relationships
        pos_emb = pos_emb.transpose(0, 1)  # (seq_len, batch_size, d_model)
        attended_pos, _ = self.temporal_attention(pos_emb, pos_emb, pos_emb)
        attended_pos = attended_pos.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Add to input and normalize
        x = x + self.dropout(attended_pos)
        x = self.norm(x)
        
        return x

# ============ Attention Mechanisms ============
class MultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with residual connections and dropout"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, 
                 attention_dropout: float = 0.1, bias: bool = True,
                 use_relative_position: bool = False, max_relative_position: int = 64):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = math.sqrt(self.d_k)
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position encoding
        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.relative_position_bias = nn.Parameter(
                torch.zeros(2 * max_relative_position - 1, nhead)
            )
            nn.init.xavier_uniform_(self.relative_position_bias)
            self.max_relative_position = max_relative_position
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Bucket relative positions for efficient computation"""
        num_buckets = self.max_relative_position
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(self.max_relative_position / max_exact) * 
            (num_buckets - max_exact)
        ).long()
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        return torch.where(is_small, relative_position, relative_position_if_large)
    
    def _compute_relative_position_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        """Compute relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        rp_bucket = self._relative_position_bucket(relative_position)
        rp_bucket = rp_bucket + self.max_relative_position - 1
        
        values = F.embedding(rp_bucket, self.relative_position_bias)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        return values
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, value_len, d_model)
            key_padding_mask: (batch_size, key_len)
            attention_mask: (query_len, key_len)
            need_weights: Whether to return attention weights
        
        Returns:
            output: (batch_size, query_len, d_model)
            attention_weights: Optional (batch_size, nhead, query_len, key_len)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add relative position bias if enabled
        if self.use_relative_position:
            relative_bias = self._compute_relative_position_bias(
                query.size(1), key.size(1)
            ).to(attention_scores.device)
            attention_scores = attention_scores + relative_bias
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Compute attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(context)
        output = self.dropout(output)
        
        if need_weights:
            return output, attention_probs
        return output, None

class ProbSparseAttention(nn.Module):
    """Probabilistic Sparse Attention for efficient long sequences"""
    
    def __init__(self, d_model: int, nhead: int, factor: int = 5, 
                 dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor
        self.temperature = temperature
        self.scale = math.sqrt(self.d_k)
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _compute_query_importance(self, Q: torch.Tensor) -> torch.Tensor:
        """Compute importance of queries using max-mean measurement"""
        # Q: (batch_size, nhead, seq_len, d_k)
        # Use max-mean as in Informer paper
        importance = torch.max(Q, dim=-1)[0] - torch.mean(Q, dim=-1)
        return importance
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Sample top-k queries based on importance
        importance = self._compute_query_importance(Q)  # (batch_size, nhead, seq_len)
        
        # Select top M queries (M = factor * log(L))
        M = self.factor * int(math.log(seq_len))
        _, top_indices = torch.topk(importance, M, dim=-1)  # (batch_size, nhead, M)
        
        # Gather selected queries
        batch_indices = torch.arange(batch_size, device=query.device)[:, None, None]
        head_indices = torch.arange(self.nhead, device=query.device)[None, :, None]
        
        Q_selected = Q[batch_indices, head_indices, top_indices, :]  # (batch_size, nhead, M, d_k)
        
        # Compute attention with selected queries
        attention_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) / self.scale
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Apply temperature scaling
        attention_scores = attention_scores / self.temperature
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)  # (batch_size, nhead, M, d_k)
        
        # Place context back to original positions
        output_full = torch.zeros_like(Q)
        output_full[batch_indices, head_indices, top_indices, :] = context
        
        # Reshape and project back
        output = output_full.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(output)
        
        if need_weights:
            return output, attention_probs
        return output, None

class TemporalAttention(nn.Module):
    """Temporal attention for capturing time dependencies"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 use_causal_mask: bool = False, lookback_window: int = None):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.use_causal_mask = use_causal_mask
        self.lookback_window = lookback_window
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Create causal mask if needed
        attention_mask = None
        if self.use_causal_mask:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            attention_mask = mask.to(x.device)
        
        # Apply lookback window if specified
        if self.lookback_window is not None:
            seq_len = x.size(1)
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - self.lookback_window)
                mask[i, :start] = False
            attention_mask = ~mask.to(x.device)
            if self.use_causal_mask:
                attention_mask = attention_mask | mask.to(x.device)
        
        # Apply attention with residual connection
        residual = x
        attended, weights = self.attention(x, x, x, attention_mask=attention_mask,
                                          need_weights=need_weights)
        x = self.norm(residual + self.dropout(attended))
        
        return x, weights

# ============ Transformer Encoder Layers ============
class TransformerEncoderLayer(nn.Module):
    """Enhanced Transformer encoder layer with multiple attention types"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu",
                 attention_type: str = "standard",  # standard, probsparse, temporal
                 layer_norm_eps: float = 1e-5,
                 use_relative_position: bool = False):
        super().__init__()
        
        # Self-attention module
        if attention_type == "standard":
            self.self_attn = MultiHeadAttention(
                d_model, nhead, dropout, 
                use_relative_position=use_relative_position
            )
        elif attention_type == "probsparse":
            self.self_attn = ProbSparseAttention(d_model, nhead, dropout=dropout)
        elif attention_type == "temporal":
            self.self_attn = TemporalAttention(d_model, nhead, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src: (batch_size, seq_len, d_model)
            src_mask: (seq_len, seq_len)
            src_key_padding_mask: (batch_size, seq_len)
            need_weights: Whether to return attention weights
        
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Optional attention weights
        """
        # Self-attention with residual connection
        src2, attention_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            attention_mask=src_mask,
            need_weights=need_weights
        )
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attention_weights

# ============ Transformer Models ============
class TimeSeriesTransformer(nn.Module):
    """Complete Transformer model for time series forecasting"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, output_dim: int, dropout: float = 0.1,
                 max_len: int = 5000, use_positional_encoding: bool = True,
                 positional_encoding_type: str = "sinusoidal",  # sinusoidal, learnable
                 attention_type: str = "standard",
                 use_causal_mask: bool = False,
                 lookback_window: Optional[int] = None,
                 use_skip_connections: bool = True,
                 use_pre_norm: bool = False):
        super().__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_skip_connections = use_skip_connections
        self.use_pre_norm = use_pre_norm
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            if positional_encoding_type == "sinusoidal":
                self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
            elif positional_encoding_type == "learnable":
                self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)
            else:
                raise ValueError(f"Unknown positional encoding type: {positional_encoding_type}")
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                attention_type=attention_type,
                use_relative_position=(i % 2 == 0)  # Alternate layers
            )
            for i in range(num_layers)
        ])
        
        # Temporal attention for final aggregation
        self.temporal_attention = TemporalAttention(
            d_model, nhead, dropout,
            use_causal_mask=use_causal_mask,
            lookback_window=lookback_window
        )
        
        # Output projection with multiple heads for different horizons
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )
        
        # Layer normalization for pre-norm architecture
        if use_pre_norm:
            self.norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional key padding mask
            return_attention: Whether to return attention weights
        
        Returns:
            output: Predictions
            attention_weights: List of attention weights from each layer (if return_attention=True)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Apply transformer encoder layers
        for i, layer in enumerate(self.encoder_layers):
            # Pre-norm architecture
            if self.use_pre_norm:
                x_norm = self.norm(x)
                x_residual, attn_weights = layer(
                    x_norm, src_mask, src_key_padding_mask,
                    need_weights=return_attention
                )
                x = x + x_residual
            else:
                # Standard architecture
                x, attn_weights = layer(
                    x, src_mask, src_key_padding_mask,
                    need_weights=return_attention
                )
            
            if return_attention and attn_weights is not None:
                attention_weights.append(attn_weights)
        
        # Apply final temporal attention
        x, final_attn_weights = self.temporal_attention(x, need_weights=return_attention)
        if return_attention and final_attn_weights is not None:
            attention_weights.append(final_attn_weights)
        
        # Global temporal pooling (use attention-weighted pooling)
        if return_attention and len(attention_weights) > 0:
            # Use attention from last layer for pooling
            last_attention = attention_weights[-1].mean(dim=1)  # Average over heads
            x_pooled = torch.bmm(last_attention, x)  # (batch_size, 1, d_model)
            x_pooled = x_pooled.squeeze(1)
        else:
            # Use mean pooling as fallback
            x_pooled = x.mean(dim=1)
        
        # Output projection
        output = self.output_projection(x_pooled)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions without returning attention weights"""
        return self.forward(x, return_attention=False)

class InformerModel(TimeSeriesTransformer):
    """Informer model with ProbSparse attention for long sequence forecasting"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, output_dim: int, dropout: float = 0.1,
                 factor: int = 5, distil_layers: int = 3):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            output_dim=output_dim,
            dropout=dropout,
            attention_type="probsparse"
        )
        
        self.factor = factor
        self.distil_layers = distil_layers
        
        # Distillation layers for Informer
        self.distil_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                stride=2
            )
            for _ in range(distil_layers)
        ])
        
        # Initialize distillation layers
        for layer in self.distil_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Distillation
        distil_outputs = []
        x_distill = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        for distil_layer in self.distil_layers:
            x_distill = distil_layer(x_distill)
            x_distill = F.relu(x_distill)
            distil_outputs.append(x_distill.transpose(1, 2))
        
        # Use distilled sequence for transformer
        x = distil_outputs[-1]
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, need_weights=return_attention)
            if return_attention and attn_weights is not None:
                attention_weights.append(attn_weights)
        
        # Output projection
        x_pooled = x.mean(dim=1)
        output = self.output_projection(x_pooled)
        
        if return_attention:
            return output, attention_weights
        return output

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for interpretable time series forecasting"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int,
                 output_dim: int, dropout: float = 0.1, 
                 static_features_dim: int = 0, future_features_dim: int = 0):
        super().__init__()
        
        self.input_dim = input_dim
        self.static_features_dim = static_features_dim
        self.future_features_dim = future_features_dim
        
        # Feature selection network
        self.feature_selection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model // 2)
            )
            for _ in range(input_dim)
        ])
        
        # Static covariate encoder
        if static_features_dim > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_features_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model)
            )
        
        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GLU(dim=-1)  # Gating mechanism
        )
        
        # Temporal processing
        self.temporal_encoder = TimeSeriesTransformer(
            input_dim=d_model,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            output_dim=d_model,
            dropout=dropout
        )
        
        # Interpretable multi-head attention
        self.interpretable_attention = MultiHeadAttention(d_model, nhead, dropout)
        
        # Output layer with quantile regression
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim * 3)  # For mean, lower, upper bounds
        )
    
    def forward(self, x: torch.Tensor, static_features: Optional[torch.Tensor] = None,
                future_features: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Historical features (batch_size, seq_len, input_dim)
            static_features: Static covariates (batch_size, static_features_dim)
            future_features: Known future features (batch_size, future_len, future_features_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with predictions and interpretability information
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature selection
        selected_features = []
        for i, feature_layer in enumerate(self.feature_selection):
            feature = x[:, :, i:i+1]  # (batch_size, seq_len, 1)
            selected = feature_layer(feature)  # (batch_size, seq_len, d_model//2)
            selected_features.append(selected)
        
        # Concatenate selected features
        x_selected = torch.cat(selected_features, dim=-1)  # (batch_size, seq_len, d_model)
        
        # Add static features if available
        if static_features is not None:
            static_encoded = self.static_encoder(static_features)  # (batch_size, d_model)
            static_encoded = static_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            x_selected = x_selected + static_encoded
        
        # Gated residual network
        residual = self.grn(x_selected)
        x_selected = x_selected + residual
        
        # Temporal processing
        temporal_features, attention_weights = self.temporal_encoder(
            x_selected, return_attention=return_attention
        )
        
        # Interpretable attention
        if return_attention:
            attended, interpretable_weights = self.interpretable_attention(
                temporal_features, temporal_features, temporal_features,
                need_weights=True
            )
        else:
            attended, interpretable_weights = self.interpretable_attention(
                temporal_features, temporal_features, temporal_features,
                need_weights=False
            ), None
        
        # Global pooling
        context = attended.mean(dim=1)
        
        # Output with uncertainty estimation
        output = self.output_layer(context)
        output = output.view(batch_size, -1, 3)  # (batch_size, output_dim, 3)
        
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
        
        if return_attention:
            result['attention_weights'] = attention_weights
            result['interpretable_weights'] = interpretable_weights
        
        return result

# ============ PyTorch Lightning Modules ============
class TransformerLightning(pl.LightningModule):
    """PyTorch Lightning module for Transformer training"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Extract configuration
        self.input_dim = config['input_dim']
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 3)
        self.dim_feedforward = config.get('dim_feedforward', 256)
        self.output_dim = config.get('output_dim', 1)
        self.dropout = config.get('dropout', 0.1)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        
        # Model type
        model_type = config.get('model_type', 'transformer')
        
        # Create model
        if model_type == 'transformer':
            self.model = TimeSeriesTransformer(
                input_dim=self.input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        elif model_type == 'informer':
            self.model = InformerModel(
                input_dim=self.input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        elif model_type == 'tft':
            self.model = TemporalFusionTransformer(
                input_dim=self.input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()
        
        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        
        # Learning rate scheduler
        self.scheduler = None
    
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
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Number of iterations for the first restart
            T_mult=2,  # Factor by which T_0 increases after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        self.scheduler = scheduler
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss'
            }
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get attention weights for interpretability"""
        with torch.no_grad():
            _, attention_weights = self.model(x, return_attention=True)
        return attention_weights
    
    def visualize_attention(self, x: torch.Tensor, layer_idx: int = -1):
        """Visualize attention weights"""
        attention_weights = self.get_attention_weights(x)
        
        if not attention_weights:
            print("No attention weights available")
            return
        
        # Get attention from specified layer
        if layer_idx < 0:
            layer_idx = len(attention_weights) + layer_idx
        
        if layer_idx >= len(attention_weights):
            print(f"Layer {layer_idx} not available")
            return
        
        attn = attention_weights[layer_idx]
        
        # Average over heads and batches
        attn_mean = attn.mean(dim=1).mean(dim=0).cpu().numpy()
        
        # Plot attention matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(attn_mean, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Attention Matrix - Layer {layer_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.show()
        
        return attn_mean

# ============ Dataset and DataLoader ============
class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 60, stride: int = 1):
        """
        Args:
            data: Input features of shape (n_samples, n_features)
            targets: Target values of shape (n_samples, n_targets)
            sequence_length: Length of input sequences
            stride: Step size between sequences
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Precompute indices
        self.indices = []
        for i in range(0, len(data) - sequence_length, stride):
            if i + sequence_length < len(targets):
                self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i:i + self.sequence_length]
        y = self.targets[i + self.sequence_length]
        return x, y

class MultiHorizonDataset(Dataset):
    """Dataset for multi-horizon forecasting"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray,
                 sequence_length: int = 60, horizon: int = 24,
                 stride: int = 1):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.stride = stride
        
        # Precompute indices
        self.indices = []
        for i in range(0, len(data) - sequence_length - horizon + 1, stride):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i:i + self.sequence_length]
        y = self.targets[i + self.sequence_length:i + self.sequence_length + self.horizon]
        return x, y

# ============ Model Factory ============
class TransformerFactory:
    """Factory class for creating transformer models"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create transformer model based on type"""
        
        model_config = {
            'input_dim': config.get('input_dim', 50),
            'd_model': config.get('d_model', 64),
            'nhead': config.get('nhead', 8),
            'num_layers': config.get('num_layers', 3),
            'dim_feedforward': config.get('dim_feedforward', 256),
            'output_dim': config.get('output_dim', 1),
            'dropout': config.get('dropout', 0.1)
        }
        
        if model_type == 'standard':
            return TimeSeriesTransformer(**model_config)
        
        elif model_type == 'informer':
            model_config['factor'] = config.get('factor', 5)
            return InformerModel(**model_config)
        
        elif model_type == 'tft':
            return TemporalFusionTransformer(**model_config)
        
        elif model_type == 'lightning':
            return TransformerLightning(model_config)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type"""
        
        defaults = {
            'standard': {
                'input_dim': 50,
                'd_model': 64,
                'nhead': 8,
                'num_layers': 3,
                'dim_feedforward': 256,
                'output_dim': 1,
                'dropout': 0.1,
                'use_positional_encoding': True,
                'attention_type': 'standard'
            },
            'informer': {
                'input_dim': 50,
                'd_model': 64,
                'nhead': 8,
                'num_layers': 3,
                'dim_feedforward': 256,
                'output_dim': 1,
                'dropout': 0.1,
                'factor': 5,
                'distil_layers': 3
            },
            'tft': {
                'input_dim': 50,
                'd_model': 64,
                'nhead': 8,
                'num_layers': 3,
                'output_dim': 1,
                'dropout': 0.1,
                'static_features_dim': 0,
                'future_features_dim': 0
            }
        }
        
        return defaults.get(model_type, defaults['standard'])

# ============ Utility Functions ============
def create_sequences(data: np.ndarray, sequence_length: int, stride: int = 1) -> np.ndarray:
    """Create sequences from time series data"""
    sequences = []
    for i in range(0, len(data) - sequence_length + 1, stride):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Any]:
    """Normalize data with different methods"""
    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (data - mean) / std
        scaler = {'mean': mean, 'std': std, 'method': method}
    
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero
        normalized = (data - min_val) / range_val
        scaler = {'min': min_val, 'max': max_val, 'method': method}
    
    elif method == 'robust':
        median = np.median(data, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1  # Avoid division by zero
        normalized = (data - median) / iqr
        scaler = {'median': median, 'iqr': iqr, 'method': method}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, scaler

def inverse_normalize(data: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    """Inverse normalize data"""
    method = scaler['method']
    
    if method == 'standard':
        return data * scaler['std'] + scaler['mean']
    elif method == 'minmax':
        return data * (scaler['max'] - scaler['min']) + scaler['min']
    elif method == 'robust':
        return data * scaler['iqr'] + scaler['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# ============ Example Usage ============
def example_usage():
    """Example usage of transformer models"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    sequence_length = 60
    horizon = 24
    
    # Create synthetic time series data
    data = np.random.randn(n_samples, n_features).cumsum(axis=0)
    targets = np.random.randn(n_samples, 1).cumsum(axis=0)
    
    # Normalize data
    data_normalized, data_scaler = normalize_data(data)
    targets_normalized, target_scaler = normalize_data(targets)
    
    # Create dataset
    dataset = TimeSeriesDataset(
        data=data_normalized,
        targets=targets_normalized,
        sequence_length=sequence_length,
        stride=1
    )
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    config = {
        'input_dim': n_features,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 256,
        'output_dim': 1,
        'dropout': 0.1,
        'learning_rate': 0.001
    }
    
    model = TransformerLightning(config)
    
    # Train model (simplified)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
    
    # Make predictions
    test_x = torch.randn(1, sequence_length, n_features)
    with torch.no_grad():
        predictions = model(test_x)
        print(f"Predictions shape: {predictions.shape}")
    
    # Get attention weights
    attention_weights = model.get_attention_weights(test_x)
    print(f"Number of attention layers: {len(attention_weights)}")
    
    return model, dataset

if __name__ == "__main__":
    print("Transformer Models for Time Series Forecasting")
    print("=" * 50)
    
    # Test different model types
    model_types = ['standard', 'informer', 'tft']
    
    for model_type in model_types:
        print(f"\nTesting {model_type.upper()} model:")
        print("-" * 30)
        
        config = TransformerFactory.get_default_config(model_type)
        config['input_dim'] = 50
        config['output_dim'] = 1
        
        try:
            model = TransformerFactory.create_model(model_type, config)
            print(f"Model created successfully")
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test forward pass
            test_input = torch.randn(1, 60, config['input_dim'])
            output = model(test_input)
            
            if isinstance(output, dict):
                print(f"Output keys: {list(output.keys())}")
                for key, value in output.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.shape}")
            else:
                print(f"Output shape: {output.shape}")
                
        except Exception as e:
            print(f"Error creating {model_type}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Transformer models ready for time series forecasting!")