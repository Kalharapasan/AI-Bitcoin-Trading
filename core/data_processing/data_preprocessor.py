"""
Data preprocessing module for Bitcoin trading AI.
Handles data cleaning, normalization, transformation, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, signal
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import VarianceThreshold
import talib
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import project modules
from config.settings import DataSettings, ModelSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Data Types and Enums ============
class DataType(str, Enum):
    """Data types for preprocessing"""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PRODUCTION = "production"

class PreprocessingStep(str, Enum):
    """Preprocessing steps"""
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    TRANSFORMATION = "transformation"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    SEQUENCE_CREATION = "sequence_creation"
    IMBALANCE_HANDLING = "imbalance_handling"

class OutlierMethod(str, Enum):
    """Methods for outlier detection"""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    MAHALANOBIS = "mahalanobis"

class ImputationMethod(str, Enum):
    """Methods for missing value imputation"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR = "linear"
    KNN = "knn"
    INTERPOLATION = "interpolation"

# ============ Configuration ============
@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    
    # Data cleaning
    remove_duplicates: bool = True
    handle_missing_values: bool = True
    missing_threshold: float = 0.3  # Remove columns with >30% missing
    outlier_handling: bool = True
    outlier_method: OutlierMethod = OutlierMethod.ZSCORE
    outlier_threshold: float = 3.0  # For z-score method
    
    # Imputation
    imputation_method: ImputationMethod = ImputationMethod.KNN
    knn_neighbors: int = 5
    interpolation_method: str = "linear"
    
    # Normalization
    normalization_method: str = "standard"  # standard, minmax, robust
    normalize_per_feature: bool = True
    scale_range: Tuple[float, float] = (0, 1)
    
    # Transformation
    log_transform: bool = True
    power_transform: bool = False
    boxcox_transform: bool = True
    differencing: bool = True
    smoothing: bool = True
    smoothing_window: int = 3
    
    # Feature engineering
    create_lags: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    create_rolling_stats: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    create_interactions: bool = True
    interaction_degree: int = 2
    
    # Feature selection
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    remove_high_correlation: bool = True
    correlation_threshold: float = 0.95
    
    # Dimensionality reduction
    use_pca: bool = False
    pca_components: Optional[int] = None
    pca_variance_threshold: float = 0.95
    
    # Sequence creation (for time series models)
    sequence_length: int = 60
    prediction_horizon: int = 1
    step_size: int = 1
    
    # Class imbalance handling
    handle_imbalance: bool = True
    imbalance_method: str = "smote"  # smote, undersample, oversample
    target_column: str = "target"
    
    # Caching
    cache_preprocessed: bool = True
    cache_dir: str = "data/cache/preprocessed"
    
    def __post_init__(self):
        """Validate configuration"""
        if self.sequence_length < 1:
            raise ValueError("Sequence length must be >= 1")
        if self.prediction_horizon < 1:
            raise ValueError("Prediction horizon must be >= 1")
        if self.step_size < 1:
            raise ValueError("Step size must be >= 1")

# ============ Base Preprocessor ============
class BasePreprocessor:
    """Base class for data preprocessing"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.scalers = {}
        self.imputers = {}
        self.pca = None
        self.feature_columns = []
        self.target_column = None
        self.fitted = False
        self.metadata = {}
        
    def fit(self, data: pd.DataFrame):
        """Fit preprocessor to data"""
        raise NotImplementedError
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor"""
        raise NotImplementedError
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data"""
        self.fit(data)
        return self.transform(data)
        
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data"""
        raise NotImplementedError
        
    def save(self, filepath: Path):
        """Save preprocessor state"""
        raise NotImplementedError
        
    def load(self, filepath: Path):
        """Load preprocessor state"""
        raise NotImplementedError

# ============ Data Cleaner ============
class DataCleaner:
    """Handles data cleaning operations"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive data cleaning"""
        self.logger.info(f"Cleaning data with shape: {data.shape}")
        
        cleaned_data = data.copy()
        
        try:
            # Step 1: Remove duplicates
            if self.config.remove_duplicates:
                initial_rows = len(cleaned_data)
                cleaned_data = self._remove_duplicates(cleaned_data)
                removed = initial_rows - len(cleaned_data)
                if removed > 0:
                    self.logger.info(f"Removed {removed} duplicate rows")
            
            # Step 2: Handle missing values
            if self.config.handle_missing_values:
                cleaned_data = self._handle_missing_values(cleaned_data)
            
            # Step 3: Detect and handle outliers
            if self.config.outlier_handling:
                cleaned_data = self._handle_outliers(cleaned_data)
            
            # Step 4: Fix data types
            cleaned_data = self._fix_data_types(cleaned_data)
            
            # Step 5: Validate data
            cleaned_data = self._validate_data(cleaned_data)
            
            self.logger.info(f"Cleaned data shape: {cleaned_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise
        
        return cleaned_data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        # Check for duplicates
        duplicates = data.duplicated()
        if duplicates.any():
            self.logger.warning(f"Found {duplicates.sum()} duplicate rows")
            data = data[~duplicates].reset_index(drop=True)
        
        # Check for duplicate indices
        duplicate_indices = data.index.duplicated()
        if duplicate_indices.any():
            self.logger.warning(f"Found {duplicate_indices.sum()} duplicate indices")
            data = data[~duplicate_indices]
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data"""
        cleaned_data = data.copy()
        
        # Calculate missing percentages
        missing_percent = cleaned_data.isnull().sum() / len(cleaned_data)
        
        # Remove columns with too many missing values
        columns_to_drop = missing_percent[missing_percent > self.config.missing_threshold].index
        if len(columns_to_drop) > 0:
            self.logger.warning(f"Dropping columns with >{self.config.missing_threshold*100}% missing: {list(columns_to_drop)}")
            cleaned_data = cleaned_data.drop(columns=columns_to_drop)
        
        # Impute remaining missing values
        if self.config.imputation_method == ImputationMethod.KNN:
            cleaned_data = self._impute_knn(cleaned_data)
        elif self.config.imputation_method == ImputationMethod.FORWARD_FILL:
            cleaned_data = cleaned_data.ffill().bfill()
        elif self.config.imputation_method == ImputationMethod.LINEAR:
            cleaned_data = cleaned_data.interpolate(method='linear')
        elif self.config.imputation_method == ImputationMethod.MEAN:
            cleaned_data = cleaned_data.fillna(cleaned_data.mean())
        elif self.config.imputation_method == ImputationMethod.MEDIAN:
            cleaned_data = cleaned_data.fillna(cleaned_data.median())
        elif self.config.imputation_method == ImputationMethod.MODE:
            cleaned_data = cleaned_data.fillna(cleaned_data.mode().iloc[0])
        else:  # Default to forward/backward fill
            cleaned_data = cleaned_data.ffill().bfill()
        
        # Check if any missing values remain
        remaining_missing = cleaned_data.isnull().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Still have {remaining_missing} missing values after imputation")
            # Fill with 0 as last resort
            cleaned_data = cleaned_data.fillna(0)
        
        return cleaned_data
    
    def _impute_knn(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using KNN"""
        try:
            from sklearn.impute import KNNImputer
            
            # Separate numeric and non-numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return data
            
            # Impute numeric columns
            imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            numeric_data = data[numeric_cols].copy()
            imputed_numeric = imputer.fit_transform(numeric_data)
            
            # Create result DataFrame
            result = data.copy()
            result[numeric_cols] = imputed_numeric
            
            # Forward fill non-numeric columns
            if len(non_numeric_cols) > 0:
                result[non_numeric_cols] = result[non_numeric_cols].ffill().bfill()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"KNN imputation failed: {str(e)}, using forward fill")
            return data.ffill().bfill()
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers"""
        cleaned_data = data.copy()
        
        # Only handle numeric columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return cleaned_data
        
        try:
            if self.config.outlier_method == OutlierMethod.ZSCORE:
                cleaned_data = self._handle_outliers_zscore(cleaned_data, numeric_cols)
            elif self.config.outlier_method == OutlierMethod.IQR:
                cleaned_data = self._handle_outliers_iqr(cleaned_data, numeric_cols)
            elif self.config.outlier_method == OutlierMethod.ISOLATION_FOREST:
                cleaned_data = self._handle_outliers_isolation_forest(cleaned_data, numeric_cols)
            elif self.config.outlier_method == OutlierMethod.LOCAL_OUTLIER_FACTOR:
                cleaned_data = self._handle_outliers_lof(cleaned_data, numeric_cols)
            elif self.config.outlier_method == OutlierMethod.MAHALANOBIS:
                cleaned_data = self._handle_outliers_mahalanobis(cleaned_data, numeric_cols)
            else:
                # Default to Z-score
                cleaned_data = self._handle_outliers_zscore(cleaned_data, numeric_cols)
                
        except Exception as e:
            self.logger.warning(f"Outlier handling failed: {str(e)}")
            # Use winsorization as fallback
            cleaned_data = self._winsorize_outliers(cleaned_data, numeric_cols)
        
        return cleaned_data
    
    def _handle_outliers_zscore(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Handle outliers using Z-score method"""
        cleaned_data = data.copy()
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(cleaned_data[col], nan_policy='omit'))
            
            # Find outliers
            outliers = z_scores > self.config.outlier_threshold
            
            if outliers.any():
                # Cap outliers at threshold
                median = cleaned_data[col].median()
                std = cleaned_data[col].std()
                upper_bound = median + self.config.outlier_threshold * std
                lower_bound = median - self.config.outlier_threshold * std
                
                cleaned_data.loc[outliers, col] = np.clip(
                    cleaned_data.loc[outliers, col],
                    lower_bound,
                    upper_bound
                )
                
                self.logger.debug(f"Capped {outliers.sum()} outliers in column {col}")
        
        return cleaned_data
    
    def _handle_outliers_iqr(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        cleaned_data = data.copy()
        
        for col in numeric_cols:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
            
            if outliers.any():
                # Cap outliers
                cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
                
                self.logger.debug(f"Capped {outliers.sum()} outliers in column {col}")
        
        return cleaned_data
    
    def _handle_outliers_isolation_forest(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Handle outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            cleaned_data = data.copy()
            numeric_data = cleaned_data[numeric_cols].copy()
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,  # Assume 10% outliers
                random_state=42
            )
            outliers = iso_forest.fit_predict(numeric_data)
            
            # Mark outliers (-1 indicates outliers)
            outlier_mask = outliers == -1
            
            if outlier_mask.any():
                # Replace outliers with median
                for col in numeric_cols:
                    median_val = cleaned_data[col].median()
                    cleaned_data.loc[outlier_mask, col] = median_val
                
                self.logger.debug(f"Replaced {outlier_mask.sum()} outliers using Isolation Forest")
        
        except Exception as e:
            self.logger.warning(f"Isolation Forest failed: {str(e)}")
            # Fall back to IQR
            cleaned_data = self._handle_outliers_iqr(data, numeric_cols)
        
        return cleaned_data
    
    def _handle_outliers_lof(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Handle outliers using Local Outlier Factor"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            cleaned_data = data.copy()
            numeric_data = cleaned_data[numeric_cols].copy()
            
            # Fit LOF
            lof = LocalOutlierFactor(
                contamination=0.1,
                novelty=False
            )
            outliers = lof.fit_predict(numeric_data)
            
            # Mark outliers (-1 indicates outliers)
            outlier_mask = outliers == -1
            
            if outlier_mask.any():
                # Replace outliers with median
                for col in numeric_cols:
                    median_val = cleaned_data[col].median()
                    cleaned_data.loc[outlier_mask, col] = median_val
                
                self.logger.debug(f"Replaced {outlier_mask.sum()} outliers using LOF")
        
        except Exception as e:
            self.logger.warning(f"LOF failed: {str(e)}")
            # Fall back to Z-score
            cleaned_data = self._handle_outliers_zscore(data, numeric_cols)
        
        return cleaned_data
    
    def _handle_outliers_mahalanobis(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Handle outliers using Mahalanobis distance"""
        try:
            cleaned_data = data.copy()
            numeric_data = cleaned_data[numeric_cols].copy()
            
            # Calculate Mahalanobis distance
            cov_matrix = np.cov(numeric_data.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            mean = np.mean(numeric_data, axis=0)
            
            mahalanobis_dist = []
            for i in range(len(numeric_data)):
                diff = numeric_data.iloc[i] - mean
                dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                mahalanobis_dist.append(dist)
            
            mahalanobis_dist = np.array(mahalanobis_dist)
            
            # Find outliers (beyond 3 standard deviations)
            threshold = np.mean(mahalanobis_dist) + 3 * np.std(mahalanobis_dist)
            outliers = mahalanobis_dist > threshold
            
            if outliers.any():
                # Replace outliers with median
                for col in numeric_cols:
                    median_val = cleaned_data[col].median()
                    cleaned_data.loc[outliers, col] = median_val
                
                self.logger.debug(f"Replaced {outliers.sum()} outliers using Mahalanobis distance")
        
        except Exception as e:
            self.logger.warning(f"Mahalanobis distance failed: {str(e)}")
            # Fall back to IQR
            cleaned_data = self._handle_outliers_iqr(data, numeric_cols)
        
        return cleaned_data
    
    def _winsorize_outliers(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Winsorize outliers (cap at percentiles)"""
        cleaned_data = data.copy()
        
        for col in numeric_cols:
            # Cap at 1st and 99th percentiles
            lower = cleaned_data[col].quantile(0.01)
            upper = cleaned_data[col].quantile(0.99)
            
            cleaned_data[col] = np.clip(cleaned_data[col], lower, upper)
        
        return cleaned_data
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix data types in DataFrame"""
        cleaned_data = data.copy()
        
        # Convert object columns to appropriate types
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype == 'object':
                try:
                    # Try to convert to datetime
                    cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce')
                    if cleaned_data[col].isnull().all():
                        # If all NaN after datetime conversion, try numeric
                        cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                except:
                    # If conversion fails, leave as object
                    pass
        
        return cleaned_data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate cleaned data"""
        cleaned_data = data.copy()
        
        # Check for infinite values
        if np.any(np.isinf(cleaned_data.select_dtypes(include=[np.number]))):
            self.logger.warning("Found infinite values, replacing with NaN")
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
            cleaned_data = cleaned_data.ffill().bfill()
        
        # Check for negative values in columns that shouldn't have them
        positive_columns = ['volume', 'price', 'close', 'open', 'high', 'low']
        for col in positive_columns:
            if col in cleaned_data.columns:
                negative_mask = cleaned_data[col] < 0
                if negative_mask.any():
                    self.logger.warning(f"Found {negative_mask.sum()} negative values in {col}, replacing with absolute values")
                    cleaned_data.loc[negative_mask, col] = cleaned_data.loc[negative_mask, col].abs()
        
        # Check for zero or near-zero variance
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_data[col].var() < 1e-10:
                self.logger.warning(f"Column {col} has near-zero variance")
        
        return cleaned_data

# ============ Data Normalizer ============
class DataNormalizer:
    """Handles data normalization and scaling"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scalers = {}
        self.feature_ranges = {}
        self.logger = get_logger(__name__)
    
    def fit(self, data: pd.DataFrame):
        """Fit normalizer to data"""
        self.logger.info("Fitting normalizer...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.config.normalize_per_feature:
                self._fit_scaler_for_column(col, data[col])
            else:
                self._fit_scaler_for_column('_all', data[numeric_cols])
                break  # Only need to fit once for all columns
    
    def _fit_scaler_for_column(self, col_name: str, data: Union[pd.Series, pd.DataFrame]):
        """Fit scaler for a specific column"""
        if self.config.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.config.normalization_method == 'minmax':
            scaler = MinMaxScaler(feature_range=self.config.scale_range)
        elif self.config.normalization_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        if isinstance(data, pd.Series):
            scaler.fit(data.values.reshape(-1, 1))
        else:
            scaler.fit(data.values)
        
        self.scalers[col_name] = scaler
        
        # Store feature range for inverse transform
        if isinstance(data, pd.Series):
            self.feature_ranges[col_name] = {
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std())
            }
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted normalizer"""
        self.logger.info("Transforming data...")
        
        transformed_data = data.copy()
        
        if self.config.normalize_per_feature:
            # Scale each column separately
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in self.scalers:
                    scaler = self.scalers[col]
                    transformed_data[col] = scaler.transform(
                        transformed_data[col].values.reshape(-1, 1)
                    ).flatten()
                elif '_all' in self.scalers:
                    # Use global scaler
                    scaler = self.scalers['_all']
                    # Need to scale all numeric columns together
                    transformed_data[numeric_cols] = scaler.transform(
                        transformed_data[numeric_cols]
                    )
                    break
        else:
            # Scale all numeric columns together
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and '_all' in self.scalers:
                scaler = self.scalers['_all']
                transformed_data[numeric_cols] = scaler.transform(
                    transformed_data[numeric_cols]
                )
        
        return transformed_data
    
    def inverse_transform(self, data: pd.DataFrame, original_columns: List[str] = None) -> pd.DataFrame:
        """Inverse transform normalized data"""
        inverse_data = data.copy()
        
        if original_columns is None:
            original_columns = list(self.scalers.keys())
        
        if self.config.normalize_per_feature:
            for col in original_columns:
                if col in self.scalers and col in inverse_data.columns:
                    scaler = self.scalers[col]
                    inverse_data[col] = scaler.inverse_transform(
                        inverse_data[col].values.reshape(-1, 1)
                    ).flatten()
        else:
            if '_all' in self.scalers:
                numeric_cols = inverse_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = self.scalers['_all']
                    inverse_data[numeric_cols] = scaler.inverse_transform(
                        inverse_data[numeric_cols]
                    )
        
        return inverse_data

# ============ Data Transformer ============
class DataTransformer:
    """Handles data transformations (log, power, differencing, etc.)"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.transformations = {}
        self.lambda_values = {}
        self.logger = get_logger(__name__)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to data"""
        self.logger.info("Applying transformations...")
        
        transformed_data = data.copy()
        
        # Apply log transform
        if self.config.log_transform:
            transformed_data = self._apply_log_transform(transformed_data)
        
        # Apply power transform
        if self.config.power_transform:
            transformed_data = self._apply_power_transform(transformed_data)
        
        # Apply Box-Cox transform
        if self.config.boxcox_transform:
            transformed_data = self._apply_boxcox_transform(transformed_data)
        
        # Apply differencing
        if self.config.differencing:
            transformed_data = self._apply_differencing(transformed_data)
        
        # Apply smoothing
        if self.config.smoothing:
            transformed_data = self._apply_smoothing(transformed_data)
        
        return transformed_data
    
    def _apply_log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transform to appropriate columns"""
        transformed_data = data.copy()
        
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Only apply log transform to positive values
            if transformed_data[col].min() > 0:
                # Check skewness to decide if log transform is helpful
                skewness = transformed_data[col].skew()
                if abs(skewness) > 0.5:  # Moderately skewed
                    transformed_data[f'log_{col}'] = np.log1p(transformed_data[col])
                    self.transformations[f'log_{col}'] = 'log'
        
        return transformed_data
    
    def _apply_power_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply power transform (Yeo-Johnson)"""
        try:
            from sklearn.preprocessing import PowerTransformer
            
            transformed_data = data.copy()
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return transformed_data
            
            # Apply power transform to each column
            for col in numeric_cols:
                try:
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    transformed_values = pt.fit_transform(transformed_data[[col]])
                    transformed_data[f'power_{col}'] = transformed_values.flatten()
                    self.transformations[f'power_{col}'] = 'power'
                    self.lambda_values[f'power_{col}'] = pt.lambdas_[0]
                except Exception as e:
                    self.logger.debug(f"Power transform failed for {col}: {str(e)}")
        
        except Exception as e:
            self.logger.warning(f"Power transform failed: {str(e)}")
        
        return transformed_data
    
    def _apply_boxcox_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Box-Cox transform"""
        transformed_data = data.copy()
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Box-Cox requires positive values
            if transformed_data[col].min() > 0:
                try:
                    transformed_values, lambda_val = stats.boxcox(transformed_data[col])
                    transformed_data[f'boxcox_{col}'] = transformed_values
                    self.transformations[f'boxcox_{col}'] = 'boxcox'
                    self.lambda_values[f'boxcox_{col}'] = lambda_val
                except Exception as e:
                    self.logger.debug(f"Box-Cox transform failed for {col}: {str(e)}")
        
        return transformed_data
    
    def _apply_differencing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply differencing to make time series stationary"""
        transformed_data = data.copy()
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # First order differencing
            transformed_data[f'diff_1_{col}'] = transformed_data[col].diff()
            
            # Second order differencing
            transformed_data[f'diff_2_{col}'] = transformed_data[f'diff_1_{col}'].diff()
            
            # Percentage change
            transformed_data[f'pct_change_{col}'] = transformed_data[col].pct_change()
            
            self.transformations[f'diff_1_{col}'] = 'diff_1'
            self.transformations[f'diff_2_{col}'] = 'diff_2'
            self.transformations[f'pct_change_{col}'] = 'pct_change'
        
        return transformed_data
    
    def _apply_smoothing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to reduce noise"""
        transformed_data = data.copy()
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Simple moving average
            for window in [3, 5, 7]:
                transformed_data[f'sma_{window}_{col}'] = (
                    transformed_data[col].rolling(window=window).mean()
                )
                self.transformations[f'sma_{window}_{col}'] = f'sma_{window}'
            
            # Exponential moving average
            for span in [5, 10, 20]:
                transformed_data[f'ema_{span}_{col}'] = (
                    transformed_data[col].ewm(span=span, adjust=False).mean()
                )
                self.transformations[f'ema_{span}_{col}'] = f'ema_{span}'
            
            # Savitzky-Golay filter
            try:
                if len(transformed_data[col]) > 11:  # Need enough data
                    transformed_data[f'sg_{col}'] = savgol_filter(
                        transformed_data[col].fillna(method='ffill'),
                        window_length=min(11, len(transformed_data[col]) // 2 * 2 + 1),
                        polyorder=2
                    )
                    self.transformations[f'sg_{col}'] = 'savgol'
            except Exception as e:
                self.logger.debug(f"Savitzky-Golay failed for {col}: {str(e)}")
        
        return transformed_data

# ============ Feature Engineer ============
class FeatureEngineer:
    """Engineers features from preprocessed data"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.lag_columns = []
        self.rolling_columns = []
        self.interaction_columns = []
        self.logger = get_logger(__name__)
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from data"""
        self.logger.info("Engineering features...")
        
        engineered_data = data.copy()
        
        # Create lag features
        if self.config.create_lags:
            engineered_data = self._create_lag_features(engineered_data)
        
        # Create rolling statistics
        if self.config.create_rolling_stats:
            engineered_data = self._create_rolling_features(engineered_data)
        
        # Create interaction features
        if self.config.create_interactions:
            engineered_data = self._create_interaction_features(engineered_data)
        
        return engineered_data
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features"""
        engineered_data = data.copy()
        numeric_cols = engineered_data.select_dtypes(include=[np.number]).columns
        
        # Determine which columns to create lags for
        # Avoid creating lags for already lagged or derived features
        base_cols = [col for col in numeric_cols 
                    if not any(x in col for x in ['lag_', 'rolling_', 'sma_', 'ema_', 'diff_', 'pct_'])]
        
        for col in base_cols:
            for lag in self.config.lag_periods:
                new_col = f'lag_{lag}_{col}'
                engineered_data[new_col] = engineered_data[col].shift(lag)
                self.lag_columns.append(new_col)
        
        return engineered_data
    
    def _create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        engineered_data = data.copy()
        numeric_cols = engineered_data.select_dtypes(include=[np.number]).columns
        
        # Determine base columns for rolling features
        base_cols = [col for col in numeric_cols 
                    if not any(x in col for x in ['rolling_', 'lag_', 'sma_', 'ema_'])]
        
        for col in base_cols:
            for window in self.config.rolling_windows:
                # Rolling mean
                mean_col = f'rolling_mean_{window}_{col}'
                engineered_data[mean_col] = engineered_data[col].rolling(window=window).mean()
                self.rolling_columns.append(mean_col)
                
                # Rolling standard deviation
                std_col = f'rolling_std_{window}_{col}'
                engineered_data[std_col] = engineered_data[col].rolling(window=window).std()
                self.rolling_columns.append(std_col)
                
                # Rolling min/max
                min_col = f'rolling_min_{window}_{col}'
                engineered_data[min_col] = engineered_data[col].rolling(window=window).min()
                self.rolling_columns.append(min_col)
                
                max_col = f'rolling_max_{window}_{col}'
                engineered_data[max_col] = engineered_data[col].rolling(window=window).max()
                self.rolling_columns.append(max_col)
                
                # Rolling median
                median_col = f'rolling_median_{window}_{col}'
                engineered_data[median_col] = engineered_data[col].rolling(window=window).median()
                self.rolling_columns.append(median_col)
                
                # Rolling quantiles
                for q in [0.25, 0.75]:
                    q_col = f'rolling_q{int(q*100)}_{window}_{col}'
                    engineered_data[q_col] = engineered_data[col].rolling(window=window).quantile(q)
                    self.rolling_columns.append(q_col)
        
        return engineered_data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between columns"""
        engineered_data = data.copy()
        numeric_cols = engineered_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return engineered_data
        
        # Limit to top correlated features to avoid explosion
        corr_matrix = engineered_data[numeric_cols].corr().abs()
        
        # Find pairs with highest correlation
        interactions = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col_i = numeric_cols[i]
                col_j = numeric_cols[j]
                corr = corr_matrix.iloc[i, j]
                
                if corr > 0.3:  # Only create interactions for somewhat correlated features
                    interactions.append((col_i, col_j, corr))
        
        # Sort by correlation and limit number of interactions
        interactions.sort(key=lambda x: x[2], reverse=True)
        interactions = interactions[:min(20, len(interactions))]  # Limit to 20 interactions
        
        for col_i, col_j, _ in interactions:
            # Multiplication interaction
            mul_col = f'{col_i}_x_{col_j}'
            engineered_data[mul_col] = engineered_data[col_i] * engineered_data[col_j]
            self.interaction_columns.append(mul_col)
            
            # Division interaction (avoid division by zero)
            if (engineered_data[col_j] != 0).all():
                div_col = f'{col_i}_div_{col_j}'
                engineered_data[div_col] = engineered_data[col_i] / (engineered_data[col_j] + 1e-8)
                self.interaction_columns.append(div_col)
            
            # Sum interaction
            sum_col = f'{col_i}_plus_{col_j}'
            engineered_data[sum_col] = engineered_data[col_i] + engineered_data[col_j]
            self.interaction_columns.append(sum_col)
            
            # Difference interaction
            diff_col = f'{col_i}_minus_{col_j}'
            engineered_data[diff_col] = engineered_data[col_i] - engineered_data[col_j]
            self.interaction_columns.append(diff_col)
        
        return engineered_data

# ============ Feature Selector ============
class FeatureSelector:
    """Selects important features and reduces dimensionality"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.selected_features = []
        self.feature_importance = {}
        self.correlation_matrix = None
        self.variance_threshold = None
        self.pca = None
        self.logger = get_logger(__name__)
    
    def select_features(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select important features from data"""
        self.logger.info("Selecting features...")
        
        selected_data = data.copy()
        numeric_cols = selected_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return selected_data
        
        # Remove low variance features
        if self.config.remove_low_variance:
            selected_data = self._remove_low_variance_features(selected_data, numeric_cols)
            numeric_cols = selected_data.select_dtypes(include=[np.number]).columns
        
        # Remove highly correlated features
        if self.config.remove_high_correlation:
            selected_data = self._remove_highly_correlated_features(selected_data, numeric_cols)
            numeric_cols = selected_data.select_dtypes(include=[np.number]).columns
        
        # Apply PCA for dimensionality reduction
        if self.config.use_pca and len(numeric_cols) > 1:
            selected_data = self._apply_pca(selected_data, numeric_cols)
        
        self.selected_features = list(selected_data.columns)
        
        self.logger.info(f"Selected {len(self.selected_features)} features")
        
        return selected_data
    
    def _remove_low_variance_features(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Remove features with low variance"""
        variances = data[numeric_cols].var()
        threshold = self.config.variance_threshold * variances.max()
        
        # Keep features with variance above threshold
        high_variance_cols = variances[variances > threshold].index
        low_variance_cols = variances[variances <= threshold].index
        
        if len(low_variance_cols) > 0:
            self.logger.info(f"Removing {len(low_variance_cols)} low-variance features")
            data = data.drop(columns=low_variance_cols)
        
        # Store variance threshold
        self.variance_threshold = threshold
        
        return data
    
    def _remove_highly_correlated_features(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Remove highly correlated features"""
        if len(numeric_cols) < 2:
            return data
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr().abs()
        self.correlation_matrix = corr_matrix
        
        # Upper triangle of correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find columns to drop
        to_drop = [
            column for column in upper_tri.columns 
            if any(upper_tri[column] > self.config.correlation_threshold)
        ]
        
        if len(to_drop) > 0:
            self.logger.info(f"Removing {len(to_drop)} highly correlated features")
            data = data.drop(columns=to_drop)
        
        return data
    
    def _apply_pca(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        try:
            # Determine number of components
            n_components = self.config.pca_components
            if n_components is None:
                # Use enough components to explain variance threshold
                temp_pca = PCA()
                temp_pca.fit(data[numeric_cols])
                cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= self.config.pca_variance_threshold) + 1
            
            # Apply PCA
            self.pca = PCA(n_components=n_components, random_state=42)
            pca_features = self.pca.fit_transform(data[numeric_cols])
            
            # Create new DataFrame with PCA features
            pca_cols = [f'pca_{i+1}' for i in range(n_components)]
            pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=data.index)
            
            # Combine with non-numeric columns
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
            result = pd.concat([data[non_numeric_cols], pca_df], axis=1)
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            self.logger.info(f"PCA: {n_components} components explain {explained_variance:.2%} of variance")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"PCA failed: {str(e)}")
            return data

# ============ Sequence Creator ============
class SequenceCreator:
    """Creates sequences for time series models"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.sequence_indices = []
        self.logger = get_logger(__name__)
    
    def create_sequences(self, features: pd.DataFrame, 
                        targets: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for time series models"""
        self.logger.info(f"Creating sequences with length {self.config.sequence_length}")
        
        # Get numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        
        if len(numeric_features) == 0:
            raise ValueError("No numeric features found for sequence creation")
        
        X_sequences = []
        y_sequences = []
        
        # Generate sequences
        for i in range(len(numeric_features) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # Feature sequence
            X_seq = numeric_features.iloc[i:i + self.config.sequence_length].values
            X_sequences.append(X_seq)
            
            # Target sequence (if provided)
            if targets is not None:
                target_idx = i + self.config.sequence_length + self.config.prediction_horizon - 1
                if target_idx < len(targets):
                    if isinstance(targets, pd.DataFrame):
                        y_seq = targets.iloc[target_idx].values
                    else:  # pd.Series
                        y_seq = targets.iloc[target_idx]
                    y_sequences.append(y_seq)
            
            # Store indices for reference
            self.sequence_indices.append({
                'start': numeric_features.index[i],
                'end': numeric_features.index[i + self.config.sequence_length - 1],
                'target_index': numeric_features.index[i + self.config.sequence_length + self.config.prediction_horizon - 1] 
                if i + self.config.sequence_length + self.config.prediction_horizon - 1 < len(numeric_features) else None
            })
        
        X = np.array(X_sequences)
        y = np.array(y_sequences) if y_sequences else None
        
        self.logger.info(f"Created {len(X)} sequences")
        
        return X, y
    
    def create_rolling_sequences(self, features: pd.DataFrame, 
                               targets: Optional[pd.DataFrame] = None,
                               stride: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences with rolling window"""
        self.logger.info(f"Creating rolling sequences with stride {stride}")
        
        numeric_features = features.select_dtypes(include=[np.number])
        
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(numeric_features) - self.config.sequence_length - self.config.prediction_horizon + 1, stride):
            X_seq = numeric_features.iloc[i:i + self.config.sequence_length].values
            X_sequences.append(X_seq)
            
            if targets is not None:
                target_idx = i + self.config.sequence_length + self.config.prediction_horizon - 1
                if target_idx < len(targets):
                    if isinstance(targets, pd.DataFrame):
                        y_seq = targets.iloc[target_idx].values
                    else:
                        y_seq = targets.iloc[target_idx]
                    y_sequences.append(y_seq)
        
        X = np.array(X_sequences)
        y = np.array(y_sequences) if y_sequences else None
        
        self.logger.info(f"Created {len(X)} rolling sequences")
        
        return X, y

# ============ Imbalance Handler ============
class ImbalanceHandler:
    """Handles class imbalance in classification problems"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.sampler = None
        self.logger = get_logger(__name__)
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance in data"""
        if not self.config.handle_imbalance:
            return X, y
        
        self.logger.info("Handling class imbalance...")
        
        # Check if this is a classification problem
        unique_classes = np.unique(y)
        if len(unique_classes) <= 1:
            self.logger.warning("Only one class found, skipping imbalance handling")
            return X, y
        
        # Check imbalance ratio
        class_counts = np.bincount(y.astype(int))
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        if imbalance_ratio < 2:  # Not significantly imbalanced
            self.logger.info(f"Imbalance ratio {imbalance_ratio:.2f} is acceptable")
            return X, y
        
        self.logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        try:
            if self.config.imbalance_method == 'smote':
                return self._apply_smote(X, y)
            elif self.config.imbalance_method == 'undersample':
                return self._apply_undersampling(X, y)
            elif self.config.imbalance_method == 'oversample':
                return self._apply_oversampling(X, y)
            else:
                self.logger.warning(f"Unknown imbalance method: {self.config.imbalance_method}")
                return X, y
                
        except Exception as e:
            self.logger.warning(f"Imbalance handling failed: {str(e)}")
            return X, y
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for oversampling"""
        try:
            from imblearn.over_sampling import SMOTE
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            self.logger.info(f"SMOTE: {len(X)} -> {len(X_resampled)} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.warning(f"SMOTE failed: {str(e)}")
            return self._apply_oversampling(X, y)
    
    def _apply_undersampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random undersampling"""
        try:
            from imblearn.under_sampling import RandomUnderSampler
            
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            self.logger.info(f"Undersampling: {len(X)} -> {len(X_resampled)} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.warning(f"Undersampling failed: {str(e)}")
            return X, y
    
    def _apply_oversampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random oversampling"""
        try:
            from imblearn.over_sampling import RandomOverSampler
            
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
            self.logger.info(f"Oversampling: {len(X)} -> {len(X_resampled)} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.warning(f"Oversampling failed: {str(e)}")
            return X, y

# ============ Main Preprocessor ============
class BitcoinDataPreprocessor(BasePreprocessor):
    """Main data preprocessor for Bitcoin trading"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        super().__init__(config)
        
        # Initialize components
        self.cleaner = DataCleaner(self.config)
        self.normalizer = DataNormalizer(self.config)
        self.transformer = DataTransformer(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.sequence_creator = SequenceCreator(self.config)
        self.imbalance_handler = ImbalanceHandler(self.config)
        
        # State tracking
        self.preprocessing_steps = []
        self.data_statistics = {}
        self.cache = {}
    
    def fit(self, data: pd.DataFrame):
        """Fit preprocessor to data"""
        self.logger.info("Fitting preprocessor...")
        
        try:
            # Step 1: Clean data
            self.preprocessing_steps.append(PreprocessingStep.CLEANING)
            cleaned_data = self.cleaner.clean(data)
            
            # Step 2: Engineer features
            self.preprocessing_steps.append(PreprocessingStep.FEATURE_SELECTION)
            engineered_data = self.feature_engineer.engineer_features(cleaned_data)
            
            # Step 3: Transform data
            self.preprocessing_steps.append(PreprocessingStep.TRANSFORMATION)
            transformed_data = self.transformer.transform(engineered_data)
            
            # Step 4: Fit normalizer
            self.preprocessing_steps.append(PreprocessingStep.NORMALIZATION)
            self.normalizer.fit(transformed_data)
            
            # Step 5: Normalize data
            normalized_data = self.normalizer.transform(transformed_data)
            
            # Step 6: Select features
            self.preprocessing_steps.append(PreprocessingStep.FEATURE_SELECTION)
            # Note: Feature selection needs target for supervised selection
            
            # Store metadata
            self._store_metadata(data, normalized_data)
            
            self.fitted = True
            self.logger.info("Preprocessor fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting preprocessor: {str(e)}")
            raise
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        self.logger.info("Transforming data...")
        
        try:
            # Apply all transformations in order
            cleaned_data = self.cleaner.clean(data)
            engineered_data = self.feature_engineer.engineer_features(cleaned_data)
            transformed_data = self.transformer.transform(engineered_data)
            normalized_data = self.normalizer.transform(transformed_data)
            
            # Select features (use already selected features from fit)
            if self.feature_selector.selected_features:
                normalized_data = normalized_data[self.feature_selector.selected_features]
            
            self.logger.info(f"Transformed data shape: {normalized_data.shape}")
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def prepare_training_data(self, features: pd.DataFrame, 
                            targets: Optional[pd.Series] = None,
                            data_type: DataType = DataType.TRAINING) -> Dict[str, Any]:
        """Prepare data for training"""
        self.logger.info(f"Preparing {data_type.value} data...")
        
        try:
            # Transform features
            processed_features = self.transform(features)
            
            # Create sequences for time series models
            X, y = None, None
            if data_type in [DataType.TRAINING, DataType.VALIDATION, DataType.TESTING]:
                X, y = self.sequence_creator.create_sequences(processed_features, targets)
                
                # Handle class imbalance for training data
                if data_type == DataType.TRAINING and y is not None and len(y.shape) == 1:
                    X, y = self.imbalance_handler.handle_imbalance(X, y)
            
            result = {
                'features': processed_features,
                'X': X,
                'y': y,
                'feature_columns': list(processed_features.columns),
                'sequence_indices': self.sequence_creator.sequence_indices,
                'data_type': data_type.value
            }
            
            # Cache if enabled
            if self.config.cache_preprocessed and data_type != DataType.PRODUCTION:
                cache_key = f"{data_type.value}_{processed_features.shape}"
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preparing {data_type.value} data: {str(e)}")
            raise
    
    def _store_metadata(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """Store metadata about the preprocessing"""
        self.metadata = {
            'original_shape': original_data.shape,
            'processed_shape': processed_data.shape,
            'preprocessing_steps': [step.value for step in self.preprocessing_steps],
            'timestamp': datetime.now().isoformat(),
            'config': {
                'missing_threshold': self.config.missing_threshold,
                'outlier_method': self.config.outlier_method.value,
                'normalization_method': self.config.normalization_method,
                'sequence_length': self.config.sequence_length
            }
        }
        
        # Store data statistics
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.data_statistics = {
                'means': processed_data[numeric_cols].mean().to_dict(),
                'stds': processed_data[numeric_cols].std().to_dict(),
                'mins': processed_data[numeric_cols].min().to_dict(),
                'maxs': processed_data[numeric_cols].max().to_dict()
            }
    
    def save(self, filepath: Path):
        """Save preprocessor state to file"""
        try:
            state = {
                'config': self.config,
                'normalizer': self.normalizer,
                'feature_selector': self.feature_selector,
                'metadata': self.metadata,
                'data_statistics': self.data_statistics,
                'feature_columns': self.feature_columns,
                'fitted': self.fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"Preprocessor saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def load(self, filepath: Path):
        """Load preprocessor state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.config = state['config']
            self.normalizer = state['normalizer']
            self.feature_selector = state['feature_selector']
            self.metadata = state['metadata']
            self.data_statistics = state['data_statistics']
            self.feature_columns = state['feature_columns']
            self.fitted = state['fitted']
            
            # Reinitialize other components with loaded config
            self.cleaner = DataCleaner(self.config)
            self.transformer = DataTransformer(self.config)
            self.feature_engineer = FeatureEngineer(self.config)
            self.sequence_creator = SequenceCreator(self.config)
            self.imbalance_handler = ImbalanceHandler(self.config)
            
            self.logger.info(f"Preprocessor loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def get_report(self) -> Dict[str, Any]:
        """Get preprocessing report"""
        report = {
            'metadata': self.metadata,
            'data_statistics': self.data_statistics,
            'preprocessing_steps': [step.value for step in self.preprocessing_steps],
            'feature_columns': self.feature_columns,
            'fitted': self.fitted,
            'cache_size': len(self.cache)
        }
        
        if hasattr(self.feature_selector, 'selected_features'):
            report['selected_features_count'] = len(self.feature_selector.selected_features)
        
        if hasattr(self.sequence_creator, 'sequence_indices'):
            report['sequences_created'] = len(self.sequence_creator.sequence_indices)
        
        return report

# ============ Factory Functions ============
def create_preprocessor(config: Optional[Dict] = None) -> BitcoinDataPreprocessor:
    """Factory function to create a preprocessor"""
    if config:
        preprocess_config = PreprocessingConfig(**config)
    else:
        preprocess_config = PreprocessingConfig()
    
    return BitcoinDataPreprocessor(preprocess_config)

def load_preprocessor_config(config_path: Path) -> PreprocessingConfig:
    """Load preprocessing configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return PreprocessingConfig(**config_dict.get('preprocessing', {}))
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {str(e)}")
        return PreprocessingConfig()

# ============ Utility Functions ============
def split_temporal_data(data: pd.DataFrame, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally (for time series)"""
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]
    
    return train_data, val_data, test_data

def create_cv_folds(data: pd.DataFrame, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create cross-validation folds for time series"""
    folds = []
    fold_size = len(data) // n_folds
    
    for i in range(n_folds - 1):
        train_end = (i + 1) * fold_size
        val_end = (i + 2) * fold_size
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        
        folds.append((train_data, val_data))
    
    # Last fold: train on all but last fold, validate on last fold
    train_data = data.iloc[:-(fold_size)]
    val_data = data.iloc[-(fold_size):]
    folds.append((train_data, val_data))
    
    return folds

def save_preprocessed_data(data: Dict[str, Any], filepath: Path):
    """Save preprocessed data to disk"""
    try:
        if filepath.suffix == '.npz':
            # Save numpy arrays
            np.savez_compressed(
                filepath,
                X=data.get('X'),
                y=data.get('y'),
                features=data.get('features').values,
                feature_columns=data.get('feature_columns'),
                metadata=json.dumps(data.get('metadata', {}))
            )
        elif filepath.suffix == '.pkl':
            # Save as pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Save as parquet for features
            features = data.get('features')
            if features is not None:
                features.to_parquet(filepath.with_suffix('.parquet'))
            
            # Save arrays separately
            if data.get('X') is not None:
                np.save(filepath.with_suffix('.X.npy'), data['X'])
            if data.get('y') is not None:
                np.save(filepath.with_suffix('.y.npy'), data['y'])
        
        logger.info(f"Preprocessed data saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {str(e)}")
        raise

def load_preprocessed_data(filepath: Path) -> Dict[str, Any]:
    """Load preprocessed data from disk"""
    try:
        if filepath.suffix == '.npz':
            loaded = np.load(filepath, allow_pickle=True)
            
            data = {
                'X': loaded['X'] if 'X' in loaded else None,
                'y': loaded['y'] if 'y' in loaded else None,
                'features': pd.DataFrame(
                    loaded['features'],
                    columns=loaded['feature_columns']
                ) if 'features' in loaded else None,
                'feature_columns': loaded['feature_columns'].tolist() if 'feature_columns' in loaded else [],
                'metadata': json.loads(loaded['metadata']) if 'metadata' in loaded else {}
            }
            
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}
            # Load from separate files
            features_path = filepath.with_suffix('.parquet')
            X_path = filepath.with_suffix('.X.npy')
            y_path = filepath.with_suffix('.y.npy')
            
            if features_path.exists():
                data['features'] = pd.read_parquet(features_path)
                data['feature_columns'] = list(data['features'].columns)
            
            if X_path.exists():
                data['X'] = np.load(X_path)
            
            if y_path.exists():
                data['y'] = np.load(y_path)
        
        logger.info(f"Preprocessed data loaded from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise

# ============ Main Execution ============
def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitcoin Trading AI - Data Preprocessing')
    parser.add_argument('--input', type=str, required=True,
                       help='Input data file path')
    parser.add_argument('--output', type=str, default='data/processed/',
                       help='Output directory for preprocessed data')
    parser.add_argument('--config', type=str, default='config/preprocessing.yaml',
                       help='Preprocessing configuration file')
    parser.add_argument('--split', action='store_true',
                       help='Split data into train/val/test')
    parser.add_argument('--create_sequences', action='store_true',
                       help='Create sequences for time series models')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            preprocess_config = load_preprocessor_config(config_path)
        else:
            preprocess_config = PreprocessingConfig()
            logger.info(f"Using default configuration, config file not found: {config_path}")
        
        # Load data
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading data from {input_path}")
        
        # Try different file formats
        if input_path.suffix == '.parquet':
            data = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            data = pd.read_csv(input_path, index_col=0, parse_dates=True)
        elif input_path.suffix == '.json':
            data = pd.read_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        print(f"Loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Create preprocessor
        preprocessor = create_preprocessor(preprocess_config.__dict__)
        
        # Fit preprocessor
        print("\nFitting preprocessor...")
        preprocessor.fit(data)
        
        # Transform data
        print("\nTransforming data...")
        processed_data = preprocessor.transform(data)
        
        print(f"Processed data shape: {processed_data.shape}")
        
        # Save processed data
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_path = output_dir / 'features.parquet'
        processed_data.to_parquet(features_path)
        print(f"Features saved to: {features_path}")
        
        # Save preprocessor
        preprocessor_path = output_dir / 'preprocessor.pkl'
        preprocessor.save(preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")
        
        # Create and save sequences if requested
        if args.create_sequences:
            print("\nCreating sequences...")
            
            # Assume last column is target for sequence creation
            if len(processed_data.columns) > 0:
                features = processed_data.iloc[:, :-1]  # All but last column
                target = processed_data.iloc[:, -1]     # Last column as target
                
                training_data = preprocessor.prepare_training_data(
                    features, target, DataType.TRAINING
                )
                
                if training_data['X'] is not None:
                    sequences_path = output_dir / 'sequences.npz'
                    save_preprocessed_data(training_data, sequences_path)
                    print(f"Sequences saved to: {sequences_path}")
                    
                    print(f"Created {len(training_data['X'])} sequences")
                    print(f"Sequence shape: {training_data['X'].shape}")
        
        # Split data if requested
        if args.split:
            print("\nSplitting data temporally...")
            train_data, val_data, test_data = split_temporal_data(processed_data)
            
            train_path = output_dir / 'train.parquet'
            val_path = output_dir / 'val.parquet'
            test_path = output_dir / 'test.parquet'
            
            train_data.to_parquet(train_path)
            val_data.to_parquet(val_path)
            test_data.to_parquet(test_path)
            
            print(f"Train data ({len(train_data)} rows) saved to: {train_path}")
            print(f"Validation data ({len(val_data)} rows) saved to: {val_path}")
            print(f"Test data ({len(test_data)} rows) saved to: {test_path}")
        
        # Generate report
        report = preprocessor.get_report()
        report_path = output_dir / 'preprocessing_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nPreprocessing report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Original data shape: {report['metadata']['original_shape']}")
        print(f"Processed data shape: {report['metadata']['processed_shape']}")
        print(f"Preprocessing steps: {len(report['preprocessing_steps'])}")
        
        if 'selected_features_count' in report:
            print(f"Selected features: {report['selected_features_count']}")
        
        if 'sequences_created' in report:
            print(f"Sequences created: {report['sequences_created']}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()