"""
Ensemble Models for Time Series Forecasting
Combines multiple models using various ensemble techniques for improved prediction accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# ============ Base Model Classes ============
class BaseForecaster(nn.Module):
    """Base class for all forecasters in ensemble"""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        return self.forward(x)
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction uncertainty (default: zeros)"""
        return torch.zeros(x.size(0), self.output_dim, device=x.device)

class EnsembleMember(BaseForecaster):
    """Wrapper for ensemble member with weight and metadata"""
    
    def __init__(self, model: BaseForecaster, weight: float = 1.0,
                 name: str = "", uncertainty_weight: float = 1.0):
        super().__init__(model.input_dim, model.output_dim)
        self.model = model
        self.weight = weight
        self.name = name
        self.uncertainty_weight = uncertainty_weight
        self.performance_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'get_uncertainty'):
            return self.model.get_uncertainty(x) * self.uncertainty_weight
        return torch.zeros(x.size(0), self.output_dim, device=x.device)
    
    def update_weight(self, new_weight: float, momentum: float = 0.9):
        """Update weight with momentum"""
        self.weight = momentum * self.weight + (1 - momentum) * new_weight
    
    def add_performance(self, metric: float):
        """Add performance metric to history"""
        self.performance_history.append(metric)
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

# ============ Ensemble Methods ============
class WeightedAverageEnsemble(nn.Module):
    """Weighted average ensemble"""
    
    def __init__(self, members: List[EnsembleMember], temperature: float = 1.0):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.temperature = temperature
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total_weight = sum(m.weight for m in self.members)
        if total_weight > 0:
            for member in self.members:
                member.weight = member.weight / total_weight
    
    def forward(self, x: torch.Tensor, return_individual: bool = False) -> Union[torch.Tensor, Tuple]:
        """Forward pass through ensemble"""
        predictions = []
        uncertainties = []
        
        for member in self.members:
            pred = member(x)
            predictions.append(pred)
            unc = member.get_uncertainty(x)
            uncertainties.append(unc)
        
        # Weighted average
        weighted_preds = torch.stack([p * m.weight for p, m in zip(predictions, self.members)], dim=0)
        ensemble_pred = torch.sum(weighted_preds, dim=0)
        
        # Weighted uncertainty (using inverse uncertainty as weight)
        uncertainty_weights = [1.0 / (unc.mean() + 1e-8) for unc in uncertainties]
        uncertainty_weights = torch.tensor(uncertainty_weights, device=x.device)
        uncertainty_weights = F.softmax(uncertainty_weights / self.temperature, dim=0)
        
        weighted_unc = torch.stack([unc * w for unc, w in zip(uncertainties, uncertainty_weights)], dim=0)
        ensemble_unc = torch.sum(weighted_unc, dim=0)
        
        if return_individual:
            return ensemble_pred, ensemble_unc, predictions, uncertainties
        return ensemble_pred
    
    def update_weights(self, errors: List[float], learning_rate: float = 0.01):
        """Update member weights based on recent errors"""
        for member, error in zip(self.members, errors):
            # Lower error -> higher weight
            new_weight = 1.0 / (error + 1e-8)
            member.update_weight(new_weight, momentum=0.9)
        
        self._normalize_weights()

class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, members: List[EnsembleMember], meta_learner: nn.Module):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.meta_learner = meta_learner
        self.input_dim = members[0].input_dim
        self.output_dim = members[0].output_dim
    
    def forward(self, x: torch.Tensor, return_individual: bool = False) -> Union[torch.Tensor, Tuple]:
        """Forward pass through stacking ensemble"""
        # Get predictions from all members
        member_preds = []
        for member in self.members:
            pred = member(x)
            member_preds.append(pred)
        
        # Stack predictions along feature dimension
        stacked_preds = torch.cat(member_preds, dim=-1)
        
        # Meta-learner makes final prediction
        final_pred = self.meta_learner(stacked_preds)
        
        if return_individual:
            return final_pred, stacked_preds, member_preds
        return final_pred
    
    def train_meta_learner(self, X_val: torch.Tensor, y_val: torch.Tensor,
                          epochs: int = 100, lr: float = 0.001):
        """Train meta-learner on validation data"""
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.meta_learner.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Get member predictions
            with torch.no_grad():
                member_preds = []
                for member in self.members:
                    pred = member(X_val)
                    member_preds.append(pred)
                stacked_preds = torch.cat(member_preds, dim=-1)
            
            # Meta-learner prediction
            final_pred = self.meta_learner(stacked_preds)
            
            # Compute loss and backpropagate
            loss = criterion(final_pred, y_val)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Meta-learner Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

class BayesianEnsemble(nn.Module):
    """Bayesian ensemble with uncertainty quantification"""
    
    def __init__(self, members: List[EnsembleMember], prior_precision: float = 1.0):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.prior_precision = prior_precision
        
        # Initialize precision parameters
        self.member_precisions = nn.ParameterList([
            nn.Parameter(torch.ones(1) * prior_precision)
            for _ in members
        ])
    
    def forward(self, x: torch.Tensor, n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Forward pass with Bayesian inference"""
        predictions = []
        uncertainties = []
        
        # Sample from each member
        for member, precision in zip(self.members, self.member_precisions):
            pred = member(x)  # (batch_size, output_dim)
            unc = member.get_uncertainty(x)  # (batch_size, output_dim)
            
            # Add Gaussian noise based on uncertainty
            std = torch.sqrt(1.0 / (precision + 1e-8) + unc)
            samples = []
            for _ in range(n_samples):
                noise = torch.randn_like(pred) * std
                samples.append(pred + noise)
            
            member_samples = torch.stack(samples, dim=0)  # (n_samples, batch_size, output_dim)
            predictions.append(member_samples)
            uncertainties.append(unc)
        
        # Combine samples
        all_samples = torch.stack(predictions, dim=0)  # (n_members, n_samples, batch_size, output_dim)
        
        # Bayesian model averaging
        precisions = torch.stack([p for p in self.member_precisions], dim=0)
        weights = F.softmax(precisions, dim=0)
        
        # Weighted average of samples
        weighted_samples = all_samples * weights.view(-1, 1, 1, 1)
        combined_samples = torch.sum(weighted_samples, dim=0)  # (n_samples, batch_size, output_dim)
        
        # Compute statistics
        mean = torch.mean(combined_samples, dim=0)
        std = torch.std(combined_samples, dim=0)
        
        # Compute credible intervals
        lower = torch.quantile(combined_samples, 0.025, dim=0)
        upper = torch.quantile(combined_samples, 0.975, dim=0)
        
        return {
            'mean': mean,
            'std': std,
            'lower': lower,
            'upper': upper,
            'samples': combined_samples,
            'member_predictions': [p.mean(dim=0) for p in predictions]
        }
    
    def update_precisions(self, validation_losses: List[float]):
        """Update precision parameters based on validation performance"""
        for i, (precision, loss) in enumerate(zip(self.member_precisions, validation_losses)):
            # Better performance -> higher precision
            new_precision = 1.0 / (loss + 1e-8)
            precision.data = 0.9 * precision.data + 0.1 * new_precision

class DynamicEnsemble(nn.Module):
    """Dynamic ensemble that selects members based on input"""
    
    def __init__(self, members: List[EnsembleMember], selector: nn.Module,
                 temperature: float = 1.0):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.selector = selector
        self.temperature = temperature
        self.input_dim = members[0].input_dim
        self.output_dim = members[0].output_dim
    
    def forward(self, x: torch.Tensor, return_selection: bool = False) -> Union[torch.Tensor, Tuple]:
        """Forward pass with dynamic member selection"""
        # Get selection weights from selector
        selection_weights = self.selector(x)  # (batch_size, n_members)
        selection_weights = F.softmax(selection_weights / self.temperature, dim=-1)
        
        # Get predictions from all members
        member_preds = []
        for member in self.members:
            pred = member(x)
            member_preds.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(member_preds, dim=1)  # (batch_size, n_members, output_dim)
        
        # Apply selection weights
        selection_weights_expanded = selection_weights.unsqueeze(-1)  # (batch_size, n_members, 1)
        ensemble_pred = torch.sum(stacked_preds * selection_weights_expanded, dim=1)
        
        if return_selection:
            return ensemble_pred, selection_weights, member_preds
        return ensemble_pred
    
    def train_selector(self, X_train: torch.Tensor, y_train: torch.Tensor,
                      X_val: torch.Tensor, y_val: torch.Tensor,
                      epochs: int = 100, lr: float = 0.001):
        """Train selector network"""
        optimizer = torch.optim.Adam(self.selector.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            # Training phase
            self.selector.train()
            optimizer.zero_grad()
            
            # Get selection weights
            selection_weights = self.selector(X_train)
            selection_weights = F.softmax(selection_weights / self.temperature, dim=-1)
            
            # Get member predictions
            with torch.no_grad():
                member_preds = []
                for member in self.members:
                    pred = member(X_train)
                    member_preds.append(pred)
                stacked_preds = torch.stack(member_preds, dim=1)
            
            # Weighted prediction
            selection_weights_expanded = selection_weights.unsqueeze(-1)
            ensemble_pred = torch.sum(stacked_preds * selection_weights_expanded, dim=1)
            
            # Compute loss
            loss = criterion(ensemble_pred, y_train)
            loss.backward()
            optimizer.step()
            
            # Validation phase
            if (epoch + 1) % 20 == 0:
                self.selector.eval()
                with torch.no_grad():
                    val_weights = self.selector(X_val)
                    val_weights = F.softmax(val_weights / self.temperature, dim=-1)
                    
                    val_member_preds = []
                    for member in self.members:
                        pred = member(X_val)
                        val_member_preds.append(pred)
                    val_stacked_preds = torch.stack(val_member_preds, dim=1)
                    
                    val_weights_expanded = val_weights.unsqueeze(-1)
                    val_ensemble_pred = torch.sum(val_stacked_preds * val_weights_expanded, dim=1)
                    val_loss = criterion(val_ensemble_pred, y_val)
                
                print(f"Selector Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {loss.item():.6f}, "
                      f"Val Loss: {val_loss.item():.6f}")

# ============ Meta-Learner Architectures ============
class SimpleMetaLearner(nn.Module):
    """Simple neural network meta-learner"""
    
    def __init__(self, n_members: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_members * output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AttentionMetaLearner(nn.Module):
    """Attention-based meta-learner"""
    
    def __init__(self, n_members: int, output_dim: int, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Embedding for each member
        self.member_embeddings = nn.Embedding(n_members, d_model)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Initialize embeddings
        nn.init.normal_(self.member_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, member_preds: torch.Tensor) -> torch.Tensor:
        """member_preds: (batch_size, n_members, output_dim)"""
        batch_size, n_members, _ = member_preds.shape
        
        # Get member embeddings
        member_ids = torch.arange(n_members, device=member_preds.device)
        member_ids = member_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_members)
        
        embeddings = self.member_embeddings(member_ids)  # (batch_size, n_members, d_model)
        
        # Reshape for attention
        embeddings = embeddings.transpose(0, 1)  # (n_members, batch_size, d_model)
        
        # Apply self-attention
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        attended = attended.transpose(0, 1)  # (batch_size, n_members, d_model)
        
        # Pool across members
        pooled = torch.mean(attended, dim=1)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_proj(pooled)  # (batch_size, output_dim)
        
        return output

class SelectorNetwork(nn.Module):
    """Network for dynamic ensemble selection"""
    
    def __init__(self, input_dim: int, n_members: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.selector_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, n_members * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_members * 2, n_members)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x.mean(dim=1))  # Pool across time dimension
        selection_logits = self.selector_head(features)
        return selection_logits

# ============ Complete Ensemble Models ============
class MLEnsembleModel(nn.Module):
    """Complete machine learning ensemble model"""
    
    def __init__(self, members: List[EnsembleMember], ensemble_method: str = "weighted",
                 temperature: float = 1.0, meta_learner: Optional[nn.Module] = None):
        super().__init__()
        self.members = members
        self.ensemble_method = ensemble_method
        self.temperature = temperature
        
        if ensemble_method == "weighted":
            self.ensemble = WeightedAverageEnsemble(members, temperature)
        elif ensemble_method == "stacking":
            if meta_learner is None:
                meta_learner = SimpleMetaLearner(len(members), members[0].output_dim)
            self.ensemble = StackingEnsemble(members, meta_learner)
        elif ensemble_method == "bayesian":
            self.ensemble = BayesianEnsemble(members)
        elif ensemble_method == "dynamic":
            if meta_learner is None:
                selector = SelectorNetwork(members[0].input_dim, len(members))
                self.ensemble = DynamicEnsemble(members, selector, temperature)
            else:
                self.ensemble = DynamicEnsemble(members, meta_learner, temperature)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict]:
        return self.ensemble(x, **kwargs)
    
    def get_member_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from all members"""
        predictions = []
        for member in self.members:
            pred = member(x)
            predictions.append(pred.detach().cpu().numpy())
        return predictions
    
    def update_ensemble_weights(self, validation_errors: List[float]):
        """Update ensemble weights based on validation performance"""
        if isinstance(self.ensemble, WeightedAverageEnsemble):
            self.ensemble.update_weights(validation_errors)
        elif isinstance(self.ensemble, BayesianEnsemble):
            self.ensemble.update_precisions(validation_errors)
    
    def save_members(self, save_dir: str):
        """Save all ensemble members"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        for i, member in enumerate(self.members):
            member_path = save_dir / f"member_{i}_{member.name}.pth"
            torch.save(member.model.state_dict(), member_path)
            print(f"Saved member {i}: {member.name} to {member_path}")
        
        # Save ensemble configuration
        config = {
            'ensemble_method': self.ensemble_method,
            'temperature': self.temperature,
            'member_names': [m.name for m in self.members],
            'member_weights': [m.weight for m in self.members]
        }
        
        config_path = save_dir / "ensemble_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
    
    def load_members(self, save_dir: str, member_models: List[nn.Module]):
        """Load ensemble members"""
        save_dir = Path(save_dir)
        config_path = save_dir / "ensemble_config.pkl"
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Load each member
        for i, (model, member_name) in enumerate(zip(member_models, config['member_names'])):
            member_path = save_dir / f"member_{i}_{member_name}.pth"
            model.load_state_dict(torch.load(member_path))
            
            # Create ensemble member
            member = EnsembleMember(
                model=model,
                weight=config['member_weights'][i],
                name=member_name
            )
            self.members.append(member)
        
        print(f"Loaded ensemble with {len(self.members)} members")

# ============ PyTorch Lightning Module ============
class EnsembleLightning(pl.LightningModule):
    """PyTorch Lightning module for ensemble training"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Extract configuration
        self.input_dim = config['input_dim']
        self.output_dim = config.get('output_dim', 1)
        self.ensemble_method = config.get('ensemble_method', 'weighted')
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.temperature = config.get('temperature', 1.0)
        
        # Create or load ensemble members
        self.members = self._create_members(config)
        
        # Create ensemble
        self.ensemble = MLEnsembleModel(
            members=self.members,
            ensemble_method=self.ensemble_method,
            temperature=self.temperature
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()
        
        # Performance tracking
        self.validation_errors = []
        self.member_performance = {i: [] for i in range(len(self.members))}
    
    def _create_members(self, config: Dict[str, Any]) -> List[EnsembleMember]:
        """Create ensemble members based on configuration"""
        members = []
        
        # Example: Create different types of models
        # In practice, you would load pre-trained models
        member_configs = config.get('member_configs', [
            {'type': 'lstm', 'name': 'LSTM', 'weight': 0.25},
            {'type': 'transformer', 'name': 'Transformer', 'weight': 0.25},
            {'type': 'cnn_lstm', 'name': 'CNN-LSTM', 'weight': 0.25},
            {'type': 'xgboost', 'name': 'XGBoost', 'weight': 0.25}
        ])
        
        for i, member_config in enumerate(member_configs):
            # Create placeholder model (in practice, load trained models)
            model = nn.Linear(self.input_dim, self.output_dim)
            
            member = EnsembleMember(
                model=model,
                weight=member_config.get('weight', 1.0 / len(member_configs)),
                name=member_config.get('name', f'Member_{i}'),
                uncertainty_weight=member_config.get('uncertainty_weight', 1.0)
            )
            members.append(member)
        
        return members
    
    def forward(self, x: torch.Tensor, **kwargs):
        return self.ensemble(x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Get ensemble prediction
        if self.ensemble_method == 'bayesian':
            output = self.ensemble(x, n_samples=50)
            y_hat = output['mean']
        else:
            y_hat = self.ensemble(x)
        
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
        
        # Get predictions from all members
        member_predictions = []
        member_errors = []
        
        for i, member in enumerate(self.members):
            with torch.no_grad():
                y_hat_member = member(x)
                member_predictions.append(y_hat_member)
                error = F.mse_loss(y_hat_member, y).item()
                member_errors.append(error)
                self.member_performance[i].append(error)
        
        # Get ensemble prediction
        if self.ensemble_method == 'bayesian':
            output = self.ensemble(x, n_samples=50)
            y_hat = output['mean']
            uncertainty = output['std'].mean().item()
            self.log('val_uncertainty', uncertainty)
        else:
            y_hat = self.ensemble(x)
        
        # Combined loss
        mse_loss = self.mse_loss(y_hat, y)
        mae_loss = self.mae_loss(y_hat, y)
        huber_loss = self.huber_loss(y_hat, y)
        loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber_loss
        
        # Store validation errors for weight updates
        self.validation_errors.append(member_errors)
        
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
        
        # Log member performances
        for i, error in enumerate(member_errors):
            self.log(f'val_member_{i}_mse', error)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Update ensemble weights at end of validation epoch"""
        if len(self.validation_errors) > 0:
            # Average errors across batches
            avg_errors = np.mean(self.validation_errors, axis=0).tolist()
            
            # Update ensemble weights
            self.ensemble.update_ensemble_weights(avg_errors)
            
            # Log updated weights
            for i, member in enumerate(self.ensemble.members):
                self.log(f'weight_member_{i}', member.weight)
            
            # Clear validation errors
            self.validation_errors.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Get ensemble prediction
        if self.ensemble_method == 'bayesian':
            output = self.ensemble(x, n_samples=100)
            y_hat = output['mean']
            uncertainty = output['std'].mean().item()
            self.log('test_uncertainty', uncertainty)
            
            # Log prediction intervals
            coverage = ((y >= output['lower']) & (y <= output['upper'])).float().mean().item()
            self.log('test_coverage_95', coverage)
            
            interval_width = (output['upper'] - output['lower']).mean().item()
            self.log('test_interval_width', interval_width)
        else:
            y_hat = self.ensemble(x)
        
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
        """Configure optimizers for trainable components"""
        
        # Only optimize components that need training (e.g., meta-learner, selector)
        params_to_optimize = []
        
        if isinstance(self.ensemble.ensemble, (StackingEnsemble, DynamicEnsemble)):
            # Add meta-learner/selector parameters
            if isinstance(self.ensemble.ensemble, StackingEnsemble):
                params_to_optimize.extend(self.ensemble.ensemble.meta_learner.parameters())
            else:  # DynamicEnsemble
                params_to_optimize.extend(self.ensemble.ensemble.selector.parameters())
        
        if len(params_to_optimize) == 0:
            # No trainable parameters
            return None
        
        optimizer = AdamW(
            params_to_optimize,
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
        return self.ensemble(x)
    
    def get_member_contributions(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Get contribution of each member to final prediction"""
        contributions = {}
        
        if isinstance(self.ensemble.ensemble, WeightedAverageEnsemble):
            for i, member in enumerate(self.ensemble.members):
                pred = member(x).detach().cpu().numpy()
                contributions[f'member_{i}_{member.name}'] = pred * member.weight
        
        elif isinstance(self.ensemble.ensemble, StackingEnsemble):
            # Get individual predictions
            member_preds = []
            for i, member in enumerate(self.ensemble.members):
                pred = member(x)
                member_preds.append(pred)
                contributions[f'member_{i}_{member.name}'] = pred.detach().cpu().numpy()
            
            # Get meta-learner weights (if available)
            if hasattr(self.ensemble.ensemble.meta_learner, 'network'):
                # Extract first layer weights as approximation
                first_layer = self.ensemble.ensemble.meta_learner.network[0]
                if isinstance(first_layer, nn.Linear):
                    weights = first_layer.weight.detach().cpu().numpy()
                    contributions['meta_learner_weights'] = weights
        
        elif isinstance(self.ensemble.ensemble, DynamicEnsemble):
            # Get selection weights
            _, selection_weights, member_preds = self.ensemble.ensemble(x, return_selection=True)
            selection_weights = selection_weights.detach().cpu().numpy()
            
            for i, (member, weight_row) in enumerate(zip(self.ensemble.members, selection_weights.T)):
                contributions[f'member_{i}_{member.name}_weight'] = weight_row
        
        return contributions
    
    def visualize_ensemble(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Visualize ensemble predictions and member contributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get predictions
        if self.ensemble_method == 'bayesian':
            output = self.ensemble(x, n_samples=100)
            ensemble_pred = output['mean'].detach().cpu().numpy()
            lower = output['lower'].detach().cpu().numpy()
            upper = output['upper'].detach().cpu().numpy()
            member_preds = output['member_predictions']
        else:
            ensemble_pred = self.ensemble(x).detach().cpu().numpy()
            member_preds = self.ensemble.get_member_predictions(x)
        
        # Plot 1: Ensemble prediction vs actual
        ax1 = axes[0, 0]
        if y is not None:
            y_np = y.detach().cpu().numpy()
            ax1.plot(y_np, label='Actual', alpha=0.7)
        ax1.plot(ensemble_pred, label='Ensemble', alpha=0.9)
        
        if self.ensemble_method == 'bayesian':
            ax1.fill_between(range(len(ensemble_pred)), lower.flatten(), upper.flatten(),
                           alpha=0.3, label='95% CI')
        
        ax1.set_title('Ensemble Prediction')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Member predictions
        ax2 = axes[0, 1]
        for i, pred in enumerate(member_preds):
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            ax2.plot(pred, alpha=0.6, label=f'Member {i}')
        
        if y is not None:
            ax2.plot(y_np, label='Actual', alpha=0.9, linewidth=2)
        
        ax2.set_title('Member Predictions')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Member weights/contributions
        ax3 = axes[1, 0]
        member_names = [m.name for m in self.ensemble.members]
        member_weights = [m.weight for m in self.ensemble.members]
        
        bars = ax3.bar(range(len(member_weights)), member_weights)
        ax3.set_xticks(range(len(member_weights)))
        ax3.set_xticklabels(member_names, rotation=45)
        ax3.set_title('Member Weights')
        ax3.set_ylabel('Weight')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Error distribution
        ax4 = axes[1, 1]
        if y is not None:
            errors = []
            for pred in member_preds:
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()
                error = np.abs(pred.flatten() - y_np.flatten())
                errors.append(error)
            
            ax4.boxplot(errors, labels=member_names)
            ax4.set_title('Member Error Distribution')
            ax4.set_ylabel('Absolute Error')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()

# ============ Ensemble Factory ============
class EnsembleFactory:
    """Factory for creating ensemble models"""
    
    @staticmethod
    def create_ensemble(ensemble_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create ensemble model based on type"""
        
        if ensemble_type == 'weighted':
            members = EnsembleFactory._create_members(config)
            return WeightedAverageEnsemble(members, config.get('temperature', 1.0))
        
        elif ensemble_type == 'stacking':
            members = EnsembleFactory._create_members(config)
            meta_learner_config = config.get('meta_learner', {'type': 'simple'})
            meta_learner = EnsembleFactory._create_meta_learner(
                len(members), members[0].output_dim, meta_learner_config
            )
            return StackingEnsemble(members, meta_learner)
        
        elif ensemble_type == 'bayesian':
            members = EnsembleFactory._create_members(config)
            return BayesianEnsemble(members, config.get('prior_precision', 1.0))
        
        elif ensemble_type == 'dynamic':
            members = EnsembleFactory._create_members(config)
            selector_config = config.get('selector', {'type': 'simple'})
            selector = EnsembleFactory._create_selector(
                members[0].input_dim, len(members), selector_config
            )
            return DynamicEnsemble(members, selector, config.get('temperature', 1.0))
        
        elif ensemble_type == 'complete':
            return MLEnsembleModel(
                members=EnsembleFactory._create_members(config),
                ensemble_method=config.get('ensemble_method', 'weighted'),
                temperature=config.get('temperature', 1.0)
            )
        
        elif ensemble_type == 'lightning':
            return EnsembleLightning(config)
        
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    @staticmethod
    def _create_members(config: Dict[str, Any]) -> List[EnsembleMember]:
        """Create ensemble members"""
        n_members = config.get('n_members', 4)
        input_dim = config.get('input_dim', 50)
        output_dim = config.get('output_dim', 1)
        
        members = []
        for i in range(n_members):
            # Create placeholder model
            model = nn.Linear(input_dim, output_dim)
            
            member = EnsembleMember(
                model=model,
                weight=1.0 / n_members,
                name=f'Member_{i}'
            )
            members.append(member)
        
        return members
    
    @staticmethod
    def _create_meta_learner(n_members: int, output_dim: int, config: Dict[str, Any]) -> nn.Module:
        """Create meta-learner"""
        meta_type = config.get('type', 'simple')
        
        if meta_type == 'simple':
            return SimpleMetaLearner(n_members, output_dim)
        elif meta_type == 'attention':
            return AttentionMetaLearner(n_members, output_dim)
        else:
            raise ValueError(f"Unknown meta-learner type: {meta_type}")
    
    @staticmethod
    def _create_selector(input_dim: int, n_members: int, config: Dict[str, Any]) -> nn.Module:
        """Create selector network"""
        selector_type = config.get('type', 'simple')
        
        if selector_type == 'simple':
            return SelectorNetwork(input_dim, n_members)
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")
    
    @staticmethod
    def get_default_config(ensemble_type: str) -> Dict[str, Any]:
        """Get default configuration for ensemble type"""
        
        defaults = {
            'weighted': {
                'input_dim': 50,
                'output_dim': 1,
                'n_members': 4,
                'temperature': 1.0,
                'learning_rate': 0.001
            },
            'stacking': {
                'input_dim': 50,
                'output_dim': 1,
                'n_members': 4,
                'meta_learner': {'type': 'simple'},
                'learning_rate': 0.001
            },
            'bayesian': {
                'input_dim': 50,
                'output_dim': 1,
                'n_members': 4,
                'prior_precision': 1.0,
                'learning_rate': 0.001
            },
            'dynamic': {
                'input_dim': 50,
                'output_dim': 1,
                'n_members': 4,
                'selector': {'type': 'simple'},
                'temperature': 1.0,
                'learning_rate': 0.001
            },
            'complete': {
                'input_dim': 50,
                'output_dim': 1,
                'ensemble_method': 'weighted',
                'temperature': 1.0,
                'learning_rate': 0.001
            }
        }
        
        return defaults.get(ensemble_type, defaults['weighted'])

# ============ Utility Functions ============
def create_diverse_ensemble(base_model_class: Any, n_members: int,
                           input_dim: int, output_dim: int,
                           config_variations: List[Dict[str, Any]]) -> List[EnsembleMember]:
    """Create diverse ensemble with different configurations"""
    members = []
    
    for i in range(n_members):
        if i < len(config_variations):
            config = config_variations[i]
        else:
            config = {}
        
        # Create model with different configuration
        model = base_model_class(input_dim=input_dim, output_dim=output_dim, **config)
        
        member = EnsembleMember(
            model=model,
            weight=1.0 / n_members,
            name=f'Member_{i}'
        )
        members.append(member)
    
    return members

def evaluate_ensemble_members(ensemble: MLEnsembleModel, dataloader: DataLoader,
                            device: str = 'cuda') -> List[float]:
    """Evaluate all ensemble members individually"""
    errors = []
    
    for member in ensemble.members:
        member.model.to(device)
        member.model.eval()
        
        total_error = 0
        n_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                y_hat = member(x)
                error = F.mse_loss(y_hat, y).item()
                total_error += error
                n_batches += 1
        
        avg_error = total_error / n_batches if n_batches > 0 else float('inf')
        errors.append(avg_error)
    
    return errors

def prune_ensemble(ensemble: MLEnsembleModel, threshold: float = 0.1) -> MLEnsembleModel:
    """Prune ensemble by removing low-performing members"""
    # Keep members with weight above threshold
    kept_members = [m for m in ensemble.members if m.weight >= threshold]
    
    if len(kept_members) == 0:
        # Keep at least one member
        kept_members = [ensemble.members[0]]
    
    # Re-normalize weights
    total_weight = sum(m.weight for m in kept_members)
    for member in kept_members:
        member.weight = member.weight / total_weight
    
    # Create new ensemble with pruned members
    pruned_ensemble = MLEnsembleModel(
        members=kept_members,
        ensemble_method=ensemble.ensemble_method,
        temperature=getattr(ensemble.ensemble, 'temperature', 1.0)
    )
    
    return pruned_ensemble

# ============ Example Usage ============
def example_usage():
    """Example usage of ensemble models"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    sequence_length = 60
    
    # Create synthetic time series data
    data = np.random.randn(n_samples, n_features).cumsum(axis=0)
    targets = np.random.randn(n_samples, 1).cumsum(axis=0)
    
    # Convert to tensors
    X = torch.FloatTensor(data[:sequence_length].reshape(1, sequence_length, n_features))
    y = torch.FloatTensor(targets[sequence_length:sequence_length+1])
    
    # Test different ensemble types
    ensemble_types = ['weighted', 'stacking', 'bayesian', 'dynamic']
    
    results = {}
    
    for ensemble_type in ensemble_types:
        print(f"\nTesting {ensemble_type} ensemble:")
        print("-" * 30)
        
        # Get default config
        config = EnsembleFactory.get_default_config(ensemble_type)
        config['input_dim'] = n_features
        config['output_dim'] = 1
        
        # Create ensemble
        ensemble = EnsembleFactory.create_ensemble(ensemble_type, config)
        
        # Test forward pass
        output = ensemble(X)
        
        if isinstance(output, dict):
            print(f"Output type: dict with keys: {list(output.keys())}")
            for key, value in output.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape}")
        else:
            print(f"Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in ensemble.parameters())
        print(f"Total parameters: {total_params:,}")
        
        results[ensemble_type] = {
            'ensemble': ensemble,
            'output': output,
            'n_params': total_params
        }
    
    return results

if __name__ == "__main__":
    print("Ensemble Models for Time Series Forecasting")
    print("=" * 50)
    
    # Test example usage
    results = example_usage()
    
    print("\n" + "=" * 50)
    print("Ensemble models are ready for time series forecasting!")
    
    # Create a complete ensemble example
    print("\nCreating complete ensemble model...")
    config = {
        'input_dim': 50,
        'output_dim': 1,
        'ensemble_method': 'weighted',
        'temperature': 1.0,
        'learning_rate': 0.001,
        'member_configs': [
            {'type': 'lstm', 'name': 'LSTM', 'weight': 0.3},
            {'type': 'transformer', 'name': 'Transformer', 'weight': 0.3},
            {'type': 'cnn_lstm', 'name': 'CNN-LSTM', 'weight': 0.2},
            {'type': 'xgboost', 'name': 'XGBoost', 'weight': 0.2}
        ]
    }
    
    complete_ensemble = MLEnsembleModel(
        members=EnsembleFactory._create_members(config),
        ensemble_method=config['ensemble_method'],
        temperature=config['temperature']
    )
    
    print(f"Complete ensemble created with {len(complete_ensemble.members)} members")
    
    # Create lightning model
    print("\nCreating PyTorch Lightning ensemble...")
    lightning_config = {
        'input_dim': 50,
        'output_dim': 1,
        'ensemble_method': 'weighted',
        'learning_rate': 0.001,
        'temperature': 1.0
    }
    
    lightning_ensemble = EnsembleLightning(lightning_config)
    print(f"Lightning ensemble created with {sum(p.numel() for p in lightning_ensemble.parameters()):,} parameters")
    
    print("\nAll ensemble models are ready!")