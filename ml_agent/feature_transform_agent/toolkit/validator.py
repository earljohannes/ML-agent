import pandas as pd

def _validate_column_exists(df: pd.DataFrame, column: str):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

def _validate_column_is_numeric(df: pd.DataFrame, column: str, operation: str):
    _validate_column_exists(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric for {operation}, but it has dtype {df[column].dtype}.")

def _validate_column_is_categorical_like(df: pd.DataFrame, column: str, operation: str):
     _validate_column_exists(df, column)
     # Check if it's category, object, or potentially boolean/low cardinality int that could be treated as categorical
     if not pd.api.types.is_categorical_dtype(df[column]) and \
        not pd.api.types.is_object_dtype(df[column]) and \
        not pd.api.types.is_bool_dtype(df[column]) and \
        not (pd.api.types.is_integer_dtype(df[column]) and df[column].nunique() < 50): # Heuristic for low-cardinality int
            print(f"Warning: Column '{column}' (dtype: {df[column].dtype}) is not typically categorical for {operation}. Proceeding, but review results.")
