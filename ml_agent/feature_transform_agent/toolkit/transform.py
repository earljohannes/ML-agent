import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import TargetEncoder
from typing import Tuple, Optional

from ml_agent.feature_transform_agent.toolkit.validator import _validate_column_exists, _validate_column_is_numeric


def implement_scale_features(df: pd.DataFrame, column: str, method: str, output_column_suffix: str) -> Tuple[pd.DataFrame, str]:
    """Applies Standard or Min-Max scaling."""
    df_copy = df.copy()
    _validate_column_is_numeric(df_copy, column, f"{method} scaling")

    values = df_copy[[column]].values # Get as numpy array
    if np.isnan(values).any():
         print(f"Warning: Column '{column}' contains NaN values. Scaling might produce NaNs.")
         # Or raise ValueError("Scaling requires non-missing values. Please impute first.")

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        # Should be caught by Pydantic Literal, but good practice
        raise ValueError(f"Unknown scaling method: {method}")

    new_col_name = f'{column}{output_column_suffix}'
    if new_col_name in df_copy.columns:
        print(f"Warning: Column '{new_col_name}' already exists. It will be overwritten.")

    df_copy[new_col_name] = scaler.fit_transform(values)
    log = f"Scaled column '{column}' using {method} scaler. New column: '{new_col_name}'."
    return df_copy, log


def implement_to_datetime(df: pd.DataFrame, column: str, datetime_format: Optional[str], coerce_errors: bool) -> Tuple[pd.DataFrame, str]:
    """Converts a column to datetime objects."""
    df_copy = df.copy()
    _validate_column_exists(df_copy, column)
    if pd.api.types.is_datetime64_any_dtype(df_copy[column]):
        print(f"Warning: Column '{column}' is already a datetime type.")
        return df_copy, f"Column '{column}' is already datetime type. No changes made."

    original_values = df_copy[column]
    errors_param = 'coerce' if coerce_errors else 'raise'

    try:
        converted_col = pd.to_datetime(original_values, format=datetime_format, errors=errors_param)
        df_copy[column] = converted_col
        log_msg = f"Converted column '{column}' to datetime objects."
        if coerce_errors and converted_col.isnull().any() and not original_values.isnull().any():
            num_nat = converted_col.isnull().sum() - original_values.isnull().sum()
            log_msg += f" {num_nat} values could not be parsed and were set to NaT."
        elif not coerce_errors and original_values.isnull().sum() != converted_col.isnull().sum():
             # This case shouldn't happen if errors='raise', just defensive
             log_msg += " Some original NaNs might remain."
    except ValueError as e:
        # This happens if errors='raise' and parsing fails
        raise ValueError(f"Failed to convert column '{column}' to datetime. Check format or data. Error: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during datetime conversion for column '{column}'. Error: {e}")

    return df_copy, log_msg
