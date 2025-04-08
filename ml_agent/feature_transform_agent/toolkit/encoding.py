import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from sklearn.preprocessing import TargetEncoder
from typing import Tuple
from category_encoders import TargetEncoder

from ml_agent.feature_transform_agent.toolkit.validator import _validate_column_exists, _validate_column_is_numeric



def _validate_column_is_categorical_like(df: pd.DataFrame, column: str, operation: str):
     _validate_column_exists(df, column)
     # Check if it's category, object, or potentially boolean/low cardinality int that could be treated as categorical
     if not pd.api.types.is_categorical_dtype(df[column]) and \
        not pd.api.types.is_object_dtype(df[column]) and \
        not pd.api.types.is_bool_dtype(df[column]) and \
        not (pd.api.types.is_integer_dtype(df[column]) and df[column].nunique() < 50): # Heuristic for low-cardinality int
            print(f"Warning: Column '{column}' (dtype: {df[column].dtype}) is not typically categorical for {operation}. Proceeding, but review results.")


def implement_one_hot_encode(df: pd.DataFrame, column: str, drop_original: bool, dummy_na: bool) -> Tuple[pd.DataFrame, str]:
    """Applies One-Hot encoding."""
    df_copy = df.copy()
    _validate_column_is_categorical_like(df_copy, column, "one-hot encoding")

    try:
        dummies = pd.get_dummies(df_copy[column], prefix=column, dummy_na=dummy_na, dtype=int)
        df_copy = pd.concat([df_copy, dummies], axis=1)
        new_cols = list(dummies.columns)
        log = f"One-hot encoded column '{column}'. New columns: {new_cols}."
        if drop_original:
            df_copy = df_copy.drop(columns=[column])
            log += f" Original column '{column}' dropped."
        else:
             log += f" Original column '{column}' kept."

    except Exception as e:
        raise RuntimeError(f"Failed to one-hot encode column '{column}'. Error: {e}")
    return df_copy, log

def implement_label_encode(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    """Applies Label encoding."""
    df_copy = df.copy()
    _validate_column_is_categorical_like(df_copy, column, "label encoding")

    encoder = LabelEncoder()
    # LabelEncoder works best on non-numeric inputs directly
    # Handle NaNs: LabelEncoder typically errors on NaNs. We can fill them or encode them separately.
    # Let's fill with a placeholder string before encoding, assuming NaN is a distinct category.
    nan_placeholder = 'NaN_placeholder_for_label_encoding'
    original_dtype = df_copy[column].dtype
    is_nan = df_copy[column].isnull()

    if is_nan.any():
        print(f"Warning: Column '{column}' contains NaN values. They will be treated as a separate category during Label Encoding.")
        # Convert to object type to hold the placeholder string if necessary
        if not pd.api.types.is_object_dtype(df_copy[column]):
             df_copy[column] = df_copy[column].astype(object)
        df_copy.loc[is_nan, column] = nan_placeholder
        encoded_values = encoder.fit_transform(df_copy[column])
        # We could potentially convert the placeholder's code back to NaN if needed
        # nan_code = encoder.transform([nan_placeholder])[0]
        # encoded_values = encoded_values.astype(float) # To allow NaN
        # encoded_values[encoded_values == nan_code] = np.nan

    else:
         encoded_values = encoder.fit_transform(df_copy[column])

    df_copy[column] = encoded_values
    log = f"Label encoded column '{column}'. Values replaced with numerical labels. Classes found: {list(encoder.classes_)}"
    return df_copy, log

def implement_target_encode(df: pd.DataFrame, column_to_encode: str, target_column: str, output_column_suffix: str, drop_original: bool) -> Tuple[pd.DataFrame, str]:
    """Applies Target encoding using category_encoders."""
    if TargetEncoder is None:
        raise RuntimeError("TargetEncoder requires the 'category_encoders' library. Please install it.")

    df_copy = df.copy()
    _validate_column_is_categorical_like(df_copy, column_to_encode, "target encoding")
    _validate_column_is_numeric(df_copy, target_column, "target encoding target")

    if df_copy[column_to_encode].isnull().any():
        print(f"Warning: Column '{column_to_encode}' contains NaN values. TargetEncoder might handle them based on its configuration (default is encoding them).")
        # Consider explicit NaN handling if needed before encoding

    if df_copy[target_column].isnull().any():
         raise ValueError(f"Target column '{target_column}' contains NaN values, which is not supported for Target Encoding. Please impute first.")

    encoder = TargetEncoder(cols=[column_to_encode], handle_missing='value', handle_unknown='value') # 'value' encodes based on overall mean

    new_col_name = f'{column_to_encode}{output_column_suffix}'
    if new_col_name in df_copy.columns:
         print(f"Warning: Column '{new_col_name}' already exists. It will be overwritten.")

    # TargetEncoder needs X (features) and y (target)
    df_copy[new_col_name] = encoder.fit_transform(df_copy[[column_to_encode]], df_copy[target_column])

    log = f"Target encoded column '{column_to_encode}' using target '{target_column}'. New column: '{new_col_name}'."

    if drop_original:
        df_copy = df_copy.drop(columns=[column_to_encode])
        log += f" Original column '{column_to_encode}' dropped."
    else:
        log += f" Original column '{column_to_encode}' kept."

    return df_copy, log
