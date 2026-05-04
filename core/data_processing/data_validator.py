"""
Data validation module for Bitcoin trading AI.
Validates data quality, integrity, and consistency for ML pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime, timedelta
from scipy import stats
import json
from pathlib import Path
import hashlib
from collections import defaultdict

# Import project modules
from config.settings import DataSettings, ModelSettings, AppConstants
from config.config_manager import get_config
from core.utils.logger import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

# ============ Data Types and Enums ============
class ValidationLevel(str, Enum):
    """Validation levels"""
    BASIC = "basic"      # Basic sanity checks
    STANDARD = "standard" # Standard validation for training
    STRICT = "strict"    # Strict validation for production
    CUSTOM = "custom"    # Custom validation rules

class ValidationStatus(str, Enum):
    """Validation status"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"

class DataIssue(str, Enum):
    """Types of data issues"""
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    INCONSISTENT_TYPES = "inconsistent_types"
    INVALID_RANGE = "invalid_range"
    INVALID_DATES = "invalid_dates"
    INCONSISTENT_FREQUENCY = "inconsistent_frequency"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SCHEMA_MISMATCH = "schema_mismatch"
    CORRUPTED_DATA = "corrupted_data"
    SEASONALITY_VIOLATION = "seasonality_violation"
    AUTOCORRELATION_VIOLATION = "autocorrelation_violation"
    STATIONARITY_VIOLATION = "stationarity_violation"

# ============ Configuration ============
@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    
    # General validation settings
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    fail_fast: bool = False
    log_details: bool = True
    generate_report: bool = True
    
    # Missing values
    check_missing_values: bool = True
    max_missing_percentage: float = 0.3
    missing_value_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Duplicates
    check_duplicates: bool = True
    check_duplicate_indices: bool = True
    check_duplicate_timestamps: bool = True
    
    # Outliers
    check_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, mahalanobis
    outlier_threshold: float = 3.0
    outlier_per_column: bool = True
    
    # Data types and ranges
    check_data_types: bool = True
    check_value_ranges: bool = True
    expected_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    expected_types: Dict[str, str] = field(default_factory=dict)
    
    # Time series specific
    check_timestamp_continuity: bool = True
    expected_frequency: Optional[str] = None  # e.g., '1H', '1D'
    max_gap_seconds: int = 3600  # 1 hour
    check_seasonality: bool = True
    check_stationarity: bool = False
    
    # Statistical properties
    check_statistical_properties: bool = True
    reference_statistics: Optional[Dict] = None
    statistical_tolerance: float = 0.2  # 20% tolerance
    
    # Data drift detection
    check_data_drift: bool = True
    drift_detection_method: str = "ks"  # ks, psi, kl_divergence
    drift_threshold: float = 0.05
    reference_data_path: Optional[str] = None
    
    # Schema validation
    check_schema: bool = True
    required_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])
    optional_columns: List[str] = field(default_factory=list)
    column_order_matters: bool = False
    
    # Business rules
    check_business_rules: bool = True
    business_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Validation caching
    cache_validation_results: bool = True
    cache_expiry_hours: int = 24
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_missing_percentage < 0 or self.max_missing_percentage > 1:
            raise ValueError("max_missing_percentage must be between 0 and 1")
        
        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")
        
        if self.statistical_tolerance < 0 or self.statistical_tolerance > 1:
            raise ValueError("statistical_tolerance must be between 0 and 1")
        
        if self.drift_threshold < 0 or self.drift_threshold > 1:
            raise ValueError("drift_threshold must be between 0 and 1")

# ============ Validation Result ============
@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

# ============ Data Quality Metrics ============
@dataclass
class DataQualityMetrics:
    """Data quality metrics"""
    completeness: float  # Percentage of non-missing values
    consistency: float   # Percentage of consistent data
    accuracy: float      # Percentage of accurate data
    timeliness: float    # Data freshness
    validity: float      # Percentage of valid values
    uniqueness: float    # Percentage of unique values
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'completeness': self.completeness,
            'consistency': self.consistency,
            'accuracy': self.accuracy,
            'timeliness': self.timeliness,
            'validity': self.validity,
            'uniqueness': self.uniqueness
        }

# ============ Base Validator ============
class BaseValidator:
    """Base class for data validators"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.results: List[ValidationResult] = []
        self.metrics: Optional[DataQualityMetrics] = None
        self.data_hash: Optional[str] = None
        self.cache_dir = Path("data/cache/validation")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data and return overall status"""
        raise NotImplementedError
    
    def get_results(self) -> List[ValidationResult]:
        """Get all validation results"""
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        status_counts = defaultdict(int)
        for result in self.results:
            status_counts[result.status.value] += 1
        
        return {
            'total_checks': len(self.results),
            'status_counts': dict(status_counts),
            'passed': status_counts['pass'],
            'warnings': status_counts['warning'],
            'failed': status_counts['fail'],
            'skipped': status_counts['skipped'],
            'overall_status': self._get_overall_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_overall_status(self) -> str:
        """Determine overall validation status"""
        if any(r.status == ValidationStatus.FAIL for r in self.results):
            return ValidationStatus.FAIL.value
        elif any(r.status == ValidationStatus.WARNING for r in self.results):
            return ValidationStatus.WARNING.value
        elif all(r.status == ValidationStatus.PASS for r in self.results):
            return ValidationStatus.PASS.value
        else:
            return ValidationStatus.SKIPPED.value
    
    def _add_result(self, result: ValidationResult):
        """Add validation result"""
        self.results.append(result)
        
        # Log based on status
        if result.status == ValidationStatus.FAIL:
            logger.error(f"Validation failed: {result.check_name} - {result.message}")
        elif result.status == ValidationStatus.WARNING:
            logger.warning(f"Validation warning: {result.check_name} - {result.message}")
        elif result.status == ValidationStatus.PASS:
            logger.debug(f"Validation passed: {result.check_name}")
        
        # Fail fast if configured
        if self.config.fail_fast and result.status == ValidationStatus.FAIL:
            raise ValueError(f"Validation failed: {result.check_name} - {result.message}")

# ============ Schema Validator ============
class SchemaValidator:
    """Validates data schema and structure"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data schema"""
        results = []
        
        # Check required columns
        if self.config.check_schema:
            results.extend(self._check_required_columns(data))
            results.extend(self._check_column_types(data))
            results.extend(self._check_column_order(data))
        
        return results
    
    def _check_required_columns(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check if all required columns are present"""
        results = []
        missing_columns = []
        
        for col in self.config.required_columns:
            if col not in data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            result = ValidationResult(
                check_name="required_columns",
                status=ValidationStatus.FAIL,
                message=f"Missing required columns: {missing_columns}",
                details={'missing_columns': missing_columns}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="required_columns",
                status=ValidationStatus.PASS,
                message="All required columns are present",
                details={'required_columns': self.config.required_columns}
            )
            results.append(result)
        
        return results
    
    def _check_column_types(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check column data types"""
        results = []
        type_mismatches = []
        
        for col, expected_type in self.config.expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if expected_type.lower() not in actual_type.lower():
                    type_mismatches.append({
                        'column': col,
                        'expected': expected_type,
                        'actual': actual_type
                    })
        
        if type_mismatches:
            result = ValidationResult(
                check_name="column_types",
                status=ValidationStatus.WARNING,
                message=f"Column type mismatches: {len(type_mismatches)}",
                details={'type_mismatches': type_mismatches}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="column_types",
                status=ValidationStatus.PASS,
                message="All column types match expectations",
                details={}
            )
            results.append(result)
        
        return results
    
    def _check_column_order(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check column order if required"""
        results = []
        
        if self.config.column_order_matters:
            expected_order = self.config.required_columns + self.config.optional_columns
            actual_order = list(data.columns)
            
            # Check if expected columns are in correct order
            order_matches = True
            for i, col in enumerate(expected_order):
                if i < len(actual_order) and actual_order[i] != col:
                    order_matches = False
                    break
            
            if not order_matches:
                result = ValidationResult(
                    check_name="column_order",
                    status=ValidationStatus.WARNING,
                    message="Column order does not match expected order",
                    details={
                        'expected_order': expected_order,
                        'actual_order': actual_order
                    }
                )
                results.append(result)
        
        return results

# ============ Completeness Validator ============
class CompletenessValidator:
    """Validates data completeness (missing values)"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data completeness"""
        results = []
        
        if self.config.check_missing_values:
            results.extend(self._check_missing_values(data))
            results.extend(self._check_missing_patterns(data))
        
        return results
    
    def _check_missing_values(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for missing values in data"""
        results = []
        
        # Calculate missing percentages
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        # Identify columns with high missing percentages
        high_missing_cols = []
        column_thresholds = []
        
        for col in data.columns:
            threshold = self.config.missing_value_thresholds.get(
                col, self.config.max_missing_percentage
            ) * 100
            
            missing_pct = missing_percentages[col]
            column_thresholds.append({
                'column': col,
                'missing_percentage': missing_pct,
                'threshold': threshold,
                'status': 'OK' if missing_pct <= threshold else 'HIGH'
            })
            
            if missing_pct > threshold:
                high_missing_cols.append({
                    'column': col,
                    'missing_percentage': missing_pct,
                    'threshold': threshold
                })
        
        if high_missing_cols:
            result = ValidationResult(
                check_name="missing_values",
                status=ValidationStatus.FAIL,
                message=f"High missing values in {len(high_missing_cols)} columns",
                details={
                    'high_missing_columns': high_missing_cols,
                    'all_columns': column_thresholds,
                    'total_missing': missing_counts.sum(),
                    'overall_missing_percentage': missing_counts.sum() / (len(data) * len(data.columns))
                }
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="missing_values",
                status=ValidationStatus.PASS,
                message="Missing values within acceptable limits",
                details={
                    'all_columns': column_thresholds,
                    'total_missing': missing_counts.sum(),
                    'overall_missing_percentage': missing_counts.sum() / (len(data) * len(data.columns))
                }
            )
            results.append(result)
        
        return results
    
    def _check_missing_patterns(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for patterns in missing data (MCAR, MAR, MNAR)"""
        results = []
        
        try:
            # Check if missingness is completely at random (MCAR)
            # by comparing means of observed and complete cases
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                patterns = []
                for col in numeric_cols[:5]:  # Limit to first 5 columns
                    missing_mask = data[col].isnull()
                    observed_mean = data[col][~missing_mask].mean()
                    
                    # If we have other columns, check correlation with missingness
                    if len(numeric_cols) > 1:
                        other_col = next(c for c in numeric_cols if c != col)
                        corr_with_missing = data[other_col].corr(missing_mask.astype(int))
                        patterns.append({
                            'column': col,
                            'missing_count': missing_mask.sum(),
                            'observed_mean': observed_mean,
                            'correlation_with_missing': corr_with_missing
                        })
                
                result = ValidationResult(
                    check_name="missing_patterns",
                    status=ValidationStatus.WARNING,
                    message="Analyzed missing data patterns",
                    details={'missing_patterns': patterns}
                )
                results.append(result)
        
        except Exception as e:
            self.logger.warning(f"Error analyzing missing patterns: {str(e)}")
        
        return results

# ============ Uniqueness Validator ============
class UniquenessValidator:
    """Validates data uniqueness (duplicates)"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate data uniqueness"""
        results = []
        
        if self.config.check_duplicates:
            results.extend(self._check_duplicate_rows(data))
            
            if self.config.check_duplicate_indices:
                results.extend(self._check_duplicate_indices(data))
            
            if self.config.check_duplicate_timestamps:
                results.extend(self._check_duplicate_timestamps(data))
        
        return results
    
    def _check_duplicate_rows(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for duplicate rows"""
        results = []
        
        # Check for duplicate rows
        duplicates = data.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            result = ValidationResult(
                check_name="duplicate_rows",
                status=ValidationStatus.FAIL,
                message=f"Found {duplicate_count} duplicate rows",
                details={
                    'duplicate_count': duplicate_count,
                    'duplicate_percentage': (duplicate_count / len(data)) * 100,
                    'duplicate_indices': data.index[duplicates].tolist()[:10]  # First 10
                }
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="duplicate_rows",
                status=ValidationStatus.PASS,
                message="No duplicate rows found",
                details={'duplicate_count': 0}
            )
            results.append(result)
        
        return results
    
    def _check_duplicate_indices(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for duplicate indices"""
        results = []
        
        # Check for duplicate indices
        duplicate_indices = data.index.duplicated()
        duplicate_index_count = duplicate_indices.sum()
        
        if duplicate_index_count > 0:
            result = ValidationResult(
                check_name="duplicate_indices",
                status=ValidationStatus.FAIL,
                message=f"Found {duplicate_index_count} duplicate indices",
                details={
                    'duplicate_index_count': duplicate_index_count,
                    'duplicate_indices': data.index[duplicate_indices].tolist()[:10]
                }
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="duplicate_indices",
                status=ValidationStatus.PASS,
                message="No duplicate indices found",
                details={'duplicate_index_count': 0}
            )
            results.append(result)
        
        return results
    
    def _check_duplicate_timestamps(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for duplicate timestamps in time series data"""
        results = []
        
        # Check if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            # Resample to check for duplicate timestamps at same frequency
            if self.config.expected_frequency:
                try:
                    resampled = data.resample(self.config.expected_frequency).count()
                    duplicate_timestamps = resampled[resampled > 1].any().any()
                    
                    if duplicate_timestamps:
                        result = ValidationResult(
                            check_name="duplicate_timestamps",
                            status=ValidationStatus.WARNING,
                            message="Potential duplicate timestamps found",
                            details={'expected_frequency': self.config.expected_frequency}
                        )
                        results.append(result)
                except Exception as e:
                    self.logger.debug(f"Error checking duplicate timestamps: {str(e)}")
        
        return results

# ============ Range Validator ============
class RangeValidator:
    """Validates value ranges and constraints"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate value ranges"""
        results = []
        
        if self.config.check_value_ranges:
            results.extend(self._check_value_ranges(data))
            results.extend(self._check_business_rules(data))
        
        return results
    
    def _check_value_ranges(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check if values are within expected ranges"""
        results = []
        range_violations = []
        
        for col, (min_val, max_val) in self.config.expected_ranges.items():
            if col in data.columns:
                # Check for values outside expected range
                below_min = data[col] < min_val
                above_max = data[col] > max_val
                
                if below_min.any() or above_max.any():
                    violations = {
                        'column': col,
                        'expected_min': min_val,
                        'expected_max': max_val,
                        'actual_min': float(data[col].min()),
                        'actual_max': float(data[col].max()),
                        'below_min_count': int(below_min.sum()),
                        'above_max_count': int(above_max.sum()),
                        'violation_indices': data.index[below_min | above_max].tolist()[:5]
                    }
                    range_violations.append(violations)
        
        if range_violations:
            result = ValidationResult(
                check_name="value_ranges",
                status=ValidationStatus.WARNING,
                message=f"Value range violations in {len(range_violations)} columns",
                details={'range_violations': range_violations}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="value_ranges",
                status=ValidationStatus.PASS,
                message="All values within expected ranges",
                details={}
            )
            results.append(result)
        
        return results
    
    def _check_business_rules(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check business rules/constraints"""
        results = []
        rule_violations = []
        
        # Common business rules for financial data
        business_rules = self.config.business_rules or {
            'high_low_rule': "high >= low",
            'open_close_range': "high >= open and high >= close and low <= open and low <= close",
            'volume_positive': "volume >= 0",
            'price_positive': "close > 0 and open > 0 and high > 0 and low > 0"
        }
        
        for rule_name, rule_expression in business_rules.items():
            try:
                # Evaluate rule
                violation_mask = ~data.eval(rule_expression)
                violation_count = violation_mask.sum()
                
                if violation_count > 0:
                    violations = {
                        'rule_name': rule_name,
                        'rule_expression': rule_expression,
                        'violation_count': int(violation_count),
                        'violation_percentage': (violation_count / len(data)) * 100,
                        'violation_indices': data.index[violation_mask].tolist()[:5]
                    }
                    rule_violations.append(violations)
            
            except Exception as e:
                self.logger.warning(f"Error evaluating rule {rule_name}: {str(e)}")
        
        if rule_violations:
            result = ValidationResult(
                check_name="business_rules",
                status=ValidationStatus.WARNING,
                message=f"Business rule violations: {len(rule_violations)} rules",
                details={'rule_violations': rule_violations}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="business_rules",
                status=ValidationStatus.PASS,
                message="All business rules satisfied",
                details={}
            )
            results.append(result)
        
        return results

# ============ Outlier Validator ============
class OutlierValidator:
    """Validates outliers in data"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate outliers in data"""
        results = []
        
        if self.config.check_outliers:
            if self.config.outlier_method == 'iqr':
                results.extend(self._check_outliers_iqr(data))
            elif self.config.outlier_method == 'zscore':
                results.extend(self._check_outliers_zscore(data))
            elif self.config.outlier_method == 'mahalanobis':
                results.extend(self._check_outliers_mahalanobis(data))
            else:
                results.extend(self._check_outliers_iqr(data))  # Default
        
        return results
    
    def _check_outliers_iqr(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check outliers using IQR method"""
        results = []
        outlier_details = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                details = {
                    'column': col,
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': (outlier_count / len(data)) * 100,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'min_value': float(data[col].min()),
                    'max_value': float(data[col].max()),
                    'outlier_indices': data.index[outliers].tolist()[:5]
                }
                outlier_details.append(details)
        
        if outlier_details:
            result = ValidationResult(
                check_name="outliers_iqr",
                status=ValidationStatus.WARNING,
                message=f"Found outliers in {len(outlier_details)} columns",
                details={'outlier_details': outlier_details}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="outliers_iqr",
                status=ValidationStatus.PASS,
                message="No significant outliers found (IQR method)",
                details={}
            )
            results.append(result)
        
        return results
    
    def _check_outliers_zscore(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check outliers using Z-score method"""
        results = []
        outlier_details = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(data[col], nan_policy='omit'))
            outliers = z_scores > self.config.outlier_threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                details = {
                    'column': col,
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': (outlier_count / len(data)) * 100,
                    'zscore_threshold': self.config.outlier_threshold,
                    'max_zscore': float(z_scores.max()),
                    'outlier_indices': data.index[outliers].tolist()[:5]
                }
                outlier_details.append(details)
        
        if outlier_details:
            result = ValidationResult(
                check_name="outliers_zscore",
                status=ValidationStatus.WARNING,
                message=f"Found outliers in {len(outlier_details)} columns",
                details={'outlier_details': outlier_details}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="outliers_zscore",
                status=ValidationStatus.PASS,
                message="No significant outliers found (Z-score method)",
                details={}
            )
            results.append(result)
        
        return results
    
    def _check_outliers_mahalanobis(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check outliers using Mahalanobis distance"""
        results = []
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Calculate Mahalanobis distance
                numeric_data = data[numeric_cols].dropna()
                cov_matrix = np.cov(numeric_data.T)
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                mean = np.mean(numeric_data, axis=0)
                
                mahalanobis_dist = []
                for i in range(len(numeric_data)):
                    diff = numeric_data.iloc[i] - mean
                    dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                    mahalanobis_dist.append(dist)
                
                mahalanobis_dist = np.array(mahalanobis_dist)
                
                # Find outliers
                threshold = np.mean(mahalanobis_dist) + self.config.outlier_threshold * np.std(mahalanobis_dist)
                outliers = mahalanobis_dist > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    result = ValidationResult(
                        check_name="outliers_mahalanobis",
                        status=ValidationStatus.WARNING,
                        message=f"Found {outlier_count} multivariate outliers",
                        details={
                            'outlier_count': int(outlier_count),
                            'threshold': float(threshold),
                            'max_distance': float(mahalanobis_dist.max())
                        }
                    )
                    results.append(result)
                else:
                    result = ValidationResult(
                        check_name="outliers_mahalanobis",
                        status=ValidationStatus.PASS,
                        message="No multivariate outliers found",
                        details={}
                    )
                    results.append(result)
        
        except Exception as e:
            self.logger.warning(f"Mahalanobis distance calculation failed: {str(e)}")
            # Fall back to IQR
            results.extend(self._check_outliers_iqr(data))
        
        return results

# ============ Time Series Validator ============
class TimeSeriesValidator:
    """Validates time series specific properties"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate time series properties"""
        results = []
        
        # Check if data has datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            if self.config.check_timestamp_continuity:
                results.extend(self._check_timestamp_continuity(data))
            
            if self.config.expected_frequency:
                results.extend(self._check_frequency_consistency(data))
            
            if self.config.check_seasonality:
                results.extend(self._check_seasonality(data))
            
            if self.config.check_stationarity:
                results.extend(self._check_stationarity(data))
        
        return results
    
    def _check_timestamp_continuity(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for gaps in time series"""
        results = []
        
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dt.total_seconds()
            
            # Identify gaps larger than threshold
            large_gaps = time_diffs > self.config.max_gap_seconds
            gap_count = large_gaps.sum()
            
            if gap_count > 0:
                gap_details = []
                gap_indices = data.index[large_gaps]
                
                for idx in gap_indices[:5]:  # First 5 gaps
                    gap_size = time_diffs.loc[idx]
                    gap_details.append({
                        'timestamp': idx.isoformat(),
                        'gap_seconds': gap_size,
                        'gap_hours': gap_size / 3600
                    })
                
                result = ValidationResult(
                    check_name="timestamp_continuity",
                    status=ValidationStatus.WARNING,
                    message=f"Found {gap_count} gaps in time series",
                    details={
                        'gap_count': int(gap_count),
                        'max_gap_seconds': float(time_diffs.max()),
                        'average_gap_seconds': float(time_diffs.mean()),
                        'gap_details': gap_details
                    }
                )
                results.append(result)
            else:
                result = ValidationResult(
                    check_name="timestamp_continuity",
                    status=ValidationStatus.PASS,
                    message="Time series is continuous",
                    details={
                        'max_gap_seconds': float(time_diffs.max()),
                        'average_gap_seconds': float(time_diffs.mean())
                    }
                )
                results.append(result)
        
        return results
    
    def _check_frequency_consistency(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check if time series has consistent frequency"""
        results = []
        
        if self.config.expected_frequency:
            try:
                # Resample to expected frequency
                expected_freq = pd.Timedelta(self.config.expected_frequency)
                
                if len(data) > 1:
                    time_diffs = data.index.to_series().diff()
                    avg_freq = time_diffs.mean()
                    
                    # Check if average frequency matches expected
                    freq_ratio = avg_freq / expected_freq
                    
                    if abs(1 - freq_ratio) > 0.1:  # 10% tolerance
                        result = ValidationResult(
                            check_name="frequency_consistency",
                            status=ValidationStatus.WARNING,
                            message=f"Frequency mismatch: expected {self.config.expected_frequency}, average {avg_freq}",
                            details={
                                'expected_frequency': self.config.expected_frequency,
                                'average_frequency': str(avg_freq),
                                'frequency_ratio': float(freq_ratio)
                            }
                        )
                        results.append(result)
                    else:
                        result = ValidationResult(
                            check_name="frequency_consistency",
                            status=ValidationStatus.PASS,
                            message=f"Frequency consistent with {self.config.expected_frequency}",
                            details={
                                'expected_frequency': self.config.expected_frequency,
                                'average_frequency': str(avg_freq)
                            }
                        )
                        results.append(result)
            
            except Exception as e:
                self.logger.warning(f"Error checking frequency consistency: {str(e)}")
        
        return results
    
    def _check_seasonality(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for seasonality in time series"""
        results = []
        
        try:
            # Use autocorrelation to detect seasonality
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                seasonality_detected = []
                for col in numeric_cols[:3]:  # Check first 3 columns
                    series = data[col].dropna()
                    
                    if len(series) > 100:
                        # Calculate autocorrelation
                        autocorr = pd.Series(series).autocorr(lag=24)  # Daily seasonality for hourly data
                        
                        if abs(autocorr) > 0.3:
                            seasonality_detected.append({
                                'column': col,
                                'autocorrelation_lag_24': autocorr,
                                'has_seasonality': True
                            })
                
                if seasonality_detected:
                    result = ValidationResult(
                        check_name="seasonality",
                        status=ValidationStatus.PASS,
                        message="Seasonality detected in time series",
                        details={'seasonality_detected': seasonality_detected}
                    )
                    results.append(result)
        
        except Exception as e:
            self.logger.debug(f"Error checking seasonality: {str(e)}")
        
        return results
    
    def _check_stationarity(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check stationarity of time series"""
        results = []
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                stationarity_results = []
                for col in numeric_cols[:2]:  # Check first 2 columns
                    series = data[col].dropna()
                    
                    if len(series) > 50:
                        # Augmented Dickey-Fuller test
                        adf_result = adfuller(series)
                        p_value = adf_result[1]
                        
                        is_stationary = p_value < 0.05
                        stationarity_results.append({
                            'column': col,
                            'p_value': p_value,
                            'is_stationary': is_stationary,
                            'test_statistic': adf_result[0],
                            'critical_values': adf_result[4]
                        })
                
                if stationarity_results:
                    non_stationary = [r for r in stationarity_results if not r['is_stationary']]
                    
                    if non_stationary:
                        result = ValidationResult(
                            check_name="stationarity",
                            status=ValidationStatus.WARNING,
                            message=f"Non-stationary series detected: {len(non_stationary)} columns",
                            details={'stationarity_results': stationarity_results}
                        )
                        results.append(result)
                    else:
                        result = ValidationResult(
                            check_name="stationarity",
                            status=ValidationStatus.PASS,
                            message="All series are stationary",
                            details={'stationarity_results': stationarity_results}
                        )
                        results.append(result)
        
        except Exception as e:
            self.logger.debug(f"Error checking stationarity: {str(e)}")
        
        return results

# ============ Statistical Validator ============
class StatisticalValidator:
    """Validates statistical properties of data"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate statistical properties"""
        results = []
        
        if self.config.check_statistical_properties:
            results.extend(self._check_basic_statistics(data))
            
            if self.config.reference_statistics:
                results.extend(self._compare_with_reference(data))
        
        return results
    
    def _check_basic_statistics(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check basic statistical properties"""
        results = []
        statistics = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats_dict = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'skewness': float(data[col].skew()),
                'kurtosis': float(data[col].kurtosis()),
                'missing_count': int(data[col].isnull().sum()),
                'unique_count': int(data[col].nunique())
            }
            statistics[col] = stats_dict
        
        result = ValidationResult(
            check_name="basic_statistics",
            status=ValidationStatus.PASS,
            message="Basic statistics calculated",
            details={'statistics': statistics}
        )
        results.append(result)
        
        return results
    
    def _compare_with_reference(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Compare statistics with reference data"""
        results = []
        deviations = []
        
        reference_stats = self.config.reference_statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_stats:
                ref_stats = reference_stats[col]
                current_stats = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'median': data[col].median()
                }
                
                col_deviations = {}
                for stat_name, current_value in current_stats.items():
                    if stat_name in ref_stats and pd.notna(current_value) and pd.notna(ref_stats[stat_name]):
                        ref_value = ref_stats[stat_name]
                        deviation = abs((current_value - ref_value) / ref_value)
                        
                        if deviation > self.config.statistical_tolerance:
                            col_deviations[stat_name] = {
                                'current': current_value,
                                'reference': ref_value,
                                'deviation': deviation,
                                'threshold': self.config.statistical_tolerance
                            }
                
                if col_deviations:
                    deviations.append({
                        'column': col,
                        'deviations': col_deviations
                    })
        
        if deviations:
            result = ValidationResult(
                check_name="statistical_comparison",
                status=ValidationStatus.WARNING,
                message=f"Statistical deviations in {len(deviations)} columns",
                details={'deviations': deviations}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="statistical_comparison",
                status=ValidationStatus.PASS,
                message="Statistics within acceptable ranges",
                details={}
            )
            results.append(result)
        
        return results

# ============ Drift Detector ============
class DriftDetector:
    """Detects data drift and concept drift"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate(self, current_data: pd.DataFrame, 
                reference_data: Optional[pd.DataFrame] = None) -> List[ValidationResult]:
        """Detect data drift"""
        results = []
        
        if self.config.check_data_drift:
            if reference_data is None and self.config.reference_data_path:
                try:
                    reference_path = Path(self.config.reference_data_path)
                    if reference_path.exists():
                        if reference_path.suffix == '.parquet':
                            reference_data = pd.read_parquet(reference_path)
                        elif reference_path.suffix == '.csv':
                            reference_data = pd.read_csv(reference_path, index_col=0)
                except Exception as e:
                    self.logger.warning(f"Error loading reference data: {str(e)}")
            
            if reference_data is not None:
                if self.config.drift_detection_method == 'ks':
                    results.extend(self._detect_drift_ks(current_data, reference_data))
                elif self.config.drift_detection_method == 'psi':
                    results.extend(self._detect_drift_psi(current_data, reference_data))
                elif self.config.drift_detection_method == 'kl_divergence':
                    results.extend(self._detect_drift_kl(current_data, reference_data))
        
        return results
    
    def _detect_drift_ks(self, current_data: pd.DataFrame, 
                        reference_data: pd.DataFrame) -> List[ValidationResult]:
        """Detect drift using Kolmogorov-Smirnov test"""
        results = []
        drift_details = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_data.columns:
                current_series = current_data[col].dropna()
                reference_series = reference_data[col].dropna()
                
                if len(current_series) > 30 and len(reference_series) > 30:
                    ks_statistic, p_value = stats.ks_2samp(current_series, reference_series)
                    
                    if p_value < self.config.drift_threshold:
                        drift_details.append({
                            'column': col,
                            'ks_statistic': ks_statistic,
                            'p_value': p_value,
                            'drift_detected': True
                        })
        
        if drift_details:
            result = ValidationResult(
                check_name="data_drift_ks",
                status=ValidationStatus.WARNING,
                message=f"Data drift detected in {len(drift_details)} columns",
                details={'drift_details': drift_details}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="data_drift_ks",
                status=ValidationStatus.PASS,
                message="No significant data drift detected",
                details={}
            )
            results.append(result)
        
        return results
    
    def _detect_drift_psi(self, current_data: pd.DataFrame, 
                         reference_data: pd.DataFrame) -> List[ValidationResult]:
        """Detect drift using Population Stability Index"""
        results = []
        drift_details = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_data.columns:
                current_series = current_data[col].dropna()
                reference_series = reference_data[col].dropna()
                
                if len(current_series) > 30 and len(reference_series) > 30:
                    # Create bins based on reference data
                    bins = np.histogram_bin_edges(reference_series, bins=10)
                    
                    # Calculate PSI
                    ref_counts, _ = np.histogram(reference_series, bins=bins)
                    curr_counts, _ = np.histogram(current_series, bins=bins)
                    
                    # Add small epsilon to avoid division by zero
                    ref_prop = ref_counts / len(reference_series) + 1e-10
                    curr_prop = curr_counts / len(current_series) + 1e-10
                    
                    psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
                    
                    if psi > 0.1:  # Common threshold for PSI
                        drift_details.append({
                            'column': col,
                            'psi': psi,
                            'drift_detected': True
                        })
        
        if drift_details:
            result = ValidationResult(
                check_name="data_drift_psi",
                status=ValidationStatus.WARNING,
                message=f"Data drift detected in {len(drift_details)} columns",
                details={'drift_details': drift_details}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="data_drift_psi",
                status=ValidationStatus.PASS,
                message="No significant data drift detected",
                details={}
            )
            results.append(result)
        
        return results
    
    def _detect_drift_kl(self, current_data: pd.DataFrame, 
                        reference_data: pd.DataFrame) -> List[ValidationResult]:
        """Detect drift using KL Divergence"""
        results = []
        drift_details = []
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_data.columns:
                current_series = current_data[col].dropna()
                reference_series = reference_data[col].dropna()
                
                if len(current_series) > 30 and len(reference_series) > 30:
                    # Create probability distributions
                    bins = np.histogram_bin_edges(
                        np.concatenate([reference_series, current_series]), 
                        bins=10
                    )
                    
                    ref_probs, _ = np.histogram(reference_series, bins=bins, density=True)
                    curr_probs, _ = np.histogram(current_series, bins=bins, density=True)
                    
                    # Add small epsilon to avoid log(0)
                    ref_probs = ref_probs + 1e-10
                    curr_probs = curr_probs + 1e-10
                    
                    # Calculate KL Divergence
                    kl_divergence = np.sum(curr_probs * np.log(curr_probs / ref_probs))
                    
                    if kl_divergence > 0.01:  # Threshold for KL divergence
                        drift_details.append({
                            'column': col,
                            'kl_divergence': kl_divergence,
                            'drift_detected': True
                        })
        
        if drift_details:
            result = ValidationResult(
                check_name="data_drift_kl",
                status=ValidationStatus.WARNING,
                message=f"Data drift detected in {len(drift_details)} columns",
                details={'drift_details': drift_details}
            )
            results.append(result)
        else:
            result = ValidationResult(
                check_name="data_drift_kl",
                status=ValidationStatus.PASS,
                message="No significant data drift detected",
                details={}
            )
            results.append(result)
        
        return results

# ============ Main Data Validator ============
class BitcoinDataValidator(BaseValidator):
    """Main data validator for Bitcoin trading"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        super().__init__(config)
        
        # Initialize validators
        self.schema_validator = SchemaValidator(self.config)
        self.completeness_validator = CompletenessValidator(self.config)
        self.uniqueness_validator = UniquenessValidator(self.config)
        self.range_validator = RangeValidator(self.config)
        self.outlier_validator = OutlierValidator(self.config)
        self.time_series_validator = TimeSeriesValidator(self.config)
        self.statistical_validator = StatisticalValidator(self.config)
        self.drift_detector = DriftDetector(self.config)
        
        # State
        self.reference_data = None
        self.data_hash_cache = {}
    
    def validate(self, data: pd.DataFrame, 
                reference_data: Optional[pd.DataFrame] = None,
                data_type: str = "current") -> bool:
        """Validate data and return overall status"""
        self.logger.info(f"Starting data validation (type: {data_type})")
        
        try:
            # Calculate data hash for caching
            data_hash = self._calculate_data_hash(data)
            self.data_hash = data_hash
            
            # Check cache for previous validation
            if self.config.cache_validation_results:
                cached_result = self._get_cached_validation(data_hash)
                if cached_result:
                    self.logger.info("Using cached validation results")
                    self.results = cached_result['results']
                    self.metrics = cached_result.get('metrics')
                    return self._get_overall_status() == ValidationStatus.PASS.value
            
            # Store reference data if provided
            if reference_data is not None:
                self.reference_data = reference_data
            
            # Run validators based on validation level
            if self.config.validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                self._run_standard_validation(data)
            
            if self.config.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                self._run_advanced_validation(data)
            
            if self.config.validation_level == ValidationLevel.STRICT:
                self._run_strict_validation(data)
            
            # Calculate data quality metrics
            self.metrics = self._calculate_data_quality_metrics(data)
            
            # Cache results if enabled
            if self.config.cache_validation_results:
                self._cache_validation_results(data_hash)
            
            # Generate report if enabled
            if self.config.generate_report:
                self._generate_validation_report(data)
            
            overall_status = self._get_overall_status()
            self.logger.info(f"Validation completed. Overall status: {overall_status}")
            
            return overall_status == ValidationStatus.PASS.value
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            raise
    
    def _run_standard_validation(self, data: pd.DataFrame):
        """Run standard validation checks"""
        self.logger.info("Running standard validation...")
        
        # Schema validation
        schema_results = self.schema_validator.validate(data)
        self.results.extend(schema_results)
        
        # Completeness validation
        completeness_results = self.completeness_validator.validate(data)
        self.results.extend(completeness_results)
        
        # Uniqueness validation
        uniqueness_results = self.uniqueness_validator.validate(data)
        self.results.extend(uniqueness_results)
        
        # Range validation
        range_results = self.range_validator.validate(data)
        self.results.extend(range_results)
    
    def _run_advanced_validation(self, data: pd.DataFrame):
        """Run advanced validation checks"""
        self.logger.info("Running advanced validation...")
        
        # Outlier validation
        outlier_results = self.outlier_validator.validate(data)
        self.results.extend(outlier_results)
        
        # Time series validation
        time_series_results = self.time_series_validator.validate(data)
        self.results.extend(time_series_results)
        
        # Statistical validation
        statistical_results = self.statistical_validator.validate(data)
        self.results.extend(statistical_results)
    
    def _run_strict_validation(self, data: pd.DataFrame):
        """Run strict validation checks"""
        self.logger.info("Running strict validation...")
        
        # Data drift detection
        if self.reference_data is not None:
            drift_results = self.drift_detector.validate(data, self.reference_data)
            self.results.extend(drift_results)
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for caching"""
        # Use a subset of data for hashing (first 1000 rows and key columns)
        if len(data) > 1000:
            sample_data = data.iloc[:1000]
        else:
            sample_data = data
        
        # Convert to string representation
        data_str = sample_data.to_string()
        
        # Calculate hash
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cached_validation(self, data_hash: str) -> Optional[Dict]:
        """Get cached validation results"""
        cache_file = self.cache_dir / f"{data_hash}.json"
        
        if cache_file.exists():
            # Check if cache is expired
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() < self.config.cache_expiry_hours * 3600:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Convert back to ValidationResult objects
                    results = []
                    for result_dict in cached_data['results']:
                        result = ValidationResult(
                            check_name=result_dict['check_name'],
                            status=ValidationStatus(result_dict['status']),
                            message=result_dict['message'],
                            details=result_dict['details'],
                            timestamp=datetime.fromisoformat(result_dict['timestamp'])
                        )
                        results.append(result)
                    
                    cached_data['results'] = results
                    
                    if cached_data.get('metrics'):
                        metrics_dict = cached_data['metrics']
                        cached_data['metrics'] = DataQualityMetrics(**metrics_dict)
                    
                    return cached_data
                except Exception as e:
                    self.logger.warning(f"Error reading cache: {str(e)}")
        
        return None
    
    def _cache_validation_results(self, data_hash: str):
        """Cache validation results"""
        cache_file = self.cache_dir / f"{data_hash}.json"
        
        cache_data = {
            'results': [r.to_dict() for r in self.results],
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'timestamp': datetime.now().isoformat(),
            'data_hash': data_hash
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Error caching results: {str(e)}")
    
    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Calculate data quality metrics"""
        # Completeness: percentage of non-missing values
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # Uniqueness: percentage of unique rows
        duplicate_rows = data.duplicated().sum()
        uniqueness = (len(data) - duplicate_rows) / len(data) if len(data) > 0 else 0
        
        # Validity: percentage of values within expected ranges
        valid_cells = total_cells
        if self.config.expected_ranges:
            for col, (min_val, max_val) in self.config.expected_ranges.items():
                if col in data.columns:
                    invalid = (data[col] < min_val) | (data[col] > max_val)
                    valid_cells -= invalid.sum()
        validity = valid_cells / total_cells if total_cells > 0 else 0
        
        # Consistency and Accuracy are harder to calculate automatically
        # For now, we'll estimate them based on other metrics
        consistency = (completeness + validity) / 2
        accuracy = validity  # Assuming valid data is accurate
        
        # Timeliness: data freshness (if we have timestamp info)
        timeliness = 1.0
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
            latest_timestamp = data.index.max()
            age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
            
            # Score based on age (0-1, where 1 is fresh)
            if age_hours < 1:
                timeliness = 1.0
            elif age_hours < 24:
                timeliness = 0.8
            elif age_hours < 168:  # 1 week
                timeliness = 0.5
            else:
                timeliness = 0.2
        
        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            validity=validity,
            uniqueness=uniqueness
        )
    
    def _generate_validation_report(self, data: pd.DataFrame):
        """Generate validation report"""
        report = {
            'summary': self.get_summary(),
            'results': [r.to_dict() for r in self.results],
            'data_info': {
                'shape': data.shape,
                'columns': list(data.columns),
                'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'index_type': str(type(data.index))
            },
            'quality_metrics': self.metrics.to_dict() if self.metrics else None,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'validation_level': self.config.validation_level.value,
                'fail_fast': self.config.fail_fast
            }
        }
        
        # Save report
        report_dir = Path("reports/validation")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved to {report_file}")
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed validation report"""
        report = self.get_summary()
        report['results'] = [r.to_dict() for r in self.results]
        
        if self.metrics:
            report['quality_metrics'] = self.metrics.to_dict()
        
        # Add issue counts by type
        issue_counts = defaultdict(int)
        for result in self.results:
            if result.status in [ValidationStatus.FAIL, ValidationStatus.WARNING]:
                # Try to determine issue type from check name
                check_name = result.check_name.lower()
                if 'missing' in check_name:
                    issue_counts[DataIssue.MISSING_VALUES.value] += 1
                elif 'duplicate' in check_name:
                    issue_counts[DataIssue.DUPLICATES.value] += 1
                elif 'outlier' in check_name:
                    issue_counts[DataIssue.OUTLIERS.value] += 1
                elif 'range' in check_name or 'business' in check_name:
                    issue_counts[DataIssue.INVALID_RANGE.value] += 1
                elif 'frequency' in check_name or 'timestamp' in check_name:
                    issue_counts[DataIssue.INCONSISTENT_FREQUENCY.value] += 1
                elif 'drift' in check_name:
                    issue_counts[DataIssue.DATA_DRIFT.value] += 1
        
        report['issue_counts'] = dict(issue_counts)
        
        return report

# ============ Factory Functions ============
def create_data_validator(config: Optional[Dict] = None) -> BitcoinDataValidator:
    """Factory function to create a data validator"""
    if config:
        validation_config = ValidationConfig(**config)
    else:
        validation_config = ValidationConfig()
    
    return BitcoinDataValidator(validation_config)

def load_validation_config(config_path: Path) -> ValidationConfig:
    """Load validation configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ValidationConfig(**config_dict.get('validation', {}))
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {str(e)}")
        return ValidationConfig()

# ============ Utility Functions ============
def validate_data_file(filepath: Path, 
                      config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """Validate data from a file"""
    validator = create_data_validator(config.__dict__ if config else None)
    
    # Load data
    if filepath.suffix == '.parquet':
        data = pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Validate
    is_valid = validator.validate(data)
    
    return {
        'is_valid': is_valid,
        'summary': validator.get_summary(),
        'detailed_report': validator.get_detailed_report(),
        'filepath': str(filepath)
    }

def compare_datasets(current_data: pd.DataFrame, 
                    reference_data: pd.DataFrame,
                    config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """Compare two datasets for consistency and drift"""
    validator = create_data_validator(config.__dict__ if config else None)
    
    # Validate current data with reference
    is_valid = validator.validate(current_data, reference_data)
    
    report = validator.get_detailed_report()
    
    # Add comparison metrics
    comparison_metrics = {
        'row_count_diff': len(current_data) - len(reference_data),
        'column_count_diff': len(current_data.columns) - len(reference_data.columns),
        'common_columns': list(set(current_data.columns) & set(reference_data.columns)),
        'unique_to_current': list(set(current_data.columns) - set(reference_data.columns)),
        'unique_to_reference': list(set(reference_data.columns) - set(current_data.columns))
    }
    
    report['comparison_metrics'] = comparison_metrics
    
    return {
        'is_valid': is_valid,
        'report': report
    }

# ============ Main Execution ============
def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitcoin Trading AI - Data Validation')
    parser.add_argument('--input', type=str, required=True,
                       help='Input data file path')
    parser.add_argument('--reference', type=str,
                       help='Reference data file path (for drift detection)')
    parser.add_argument('--config', type=str, default='config/validation.yaml',
                       help='Validation configuration file')
    parser.add_argument('--level', type=str, choices=['basic', 'standard', 'strict'],
                       help='Validation level (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            validation_config = load_validation_config(config_path)
        else:
            validation_config = ValidationConfig()
            logger.info(f"Using default configuration, config file not found: {config_path}")
        
        # Override validation level if specified
        if args.level:
            validation_config.validation_level = ValidationLevel(args.level)
        
        # Load input data
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading data from {input_path}")
        
        if input_path.suffix == '.parquet':
            data = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            data = pd.read_csv(input_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        print(f"Loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Load reference data if provided
        reference_data = None
        if args.reference:
            reference_path = Path(args.reference)
            if reference_path.exists():
                if reference_path.suffix == '.parquet':
                    reference_data = pd.read_parquet(reference_path)
                elif reference_path.suffix == '.csv':
                    reference_data = pd.read_csv(reference_path, index_col=0, parse_dates=True)
                print(f"Loaded reference data with shape: {reference_data.shape}")
        
        # Create validator
        validator = create_data_validator(validation_config.__dict__)
        
        # Run validation
        print(f"\nRunning {validation_config.validation_level.value} validation...")
        is_valid = validator.validate(data, reference_data)
        
        # Get results
        summary = validator.get_summary()
        detailed_report = validator.get_detailed_report()
        
        # Print summary
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Overall status: {summary['overall_status'].upper()}")
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Failed: {summary['failed']}")
        
        if detailed_report.get('quality_metrics'):
            print("\nData Quality Metrics:")
            metrics = detailed_report['quality_metrics']
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.2%}")
        
        # Print failed checks
        failed_checks = [r for r in validator.results if r.status == ValidationStatus.FAIL]
        if failed_checks:
            print("\nFAILED CHECKS:")
            for check in failed_checks:
                print(f"  • {check.check_name}: {check.message}")
        
        # Print warnings
        warning_checks = [r for r in validator.results if r.status == ValidationStatus.WARNING]
        if warning_checks:
            print("\nWARNINGS:")
            for check in warning_checks[:5]:  # First 5 warnings
                print(f"  • {check.check_name}: {check.message}")
            if len(warning_checks) > 5:
                print(f"  ... and {len(warning_checks) - 5} more warnings")
        
        # Save report if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_dir / f"validation_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(detailed_report, f, indent=2, default=str)
            
            print(f"\nDetailed report saved to: {report_file}")
        
        print("\n" + "="*50)
        print(f"Validation {'PASSED' if is_valid else 'FAILED'}")
        print("="*50)
        
        # Exit with appropriate code
        import sys
        sys.exit(0 if is_valid else 1)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()