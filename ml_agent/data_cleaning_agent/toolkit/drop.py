import pandas as pd
import numpy as np
from typing import Dict, Any


def get_null_value_counts(df: pd.DataFrame) -> dict:
    """Returns percent of null values for each column in dataframe"""
    null_vals = {}

    for col in df:
        missing_vals = df[(df[col].isna() == True) | (df[col] == "")]
        nans = 100 * np.round(len(missing_vals) / len(df), 2)
        null_vals[col] = float(nans)

    return null_vals


def drop_high_null_columns(df: pd.DataFrame, null_threshold_percent: float = 50.0) -> Dict[str, Any]:
    """
    Drops columns from the DataFrame where the percentage of missing values
    (NaN or empty string) exceeds the specified threshold.
    Args:
        df (pd.DataFrame): The input DataFrame.
        null_threshold_percent (float): The percentage threshold (0-100). Default is 50.
    Returns:
        Dict[str, Any]: A dictionary containing the modified DataFrame under the key 'dataframe'
                        and a status message under the key 'status'.
    """
    result = {"status": "", "series": None, "error": None}
    print(f"--- Calling drop_high_null_columns (threshold: {null_threshold_percent}%) ---")
    if df is None:
         return {"dataframe": None, "status": "Error: No DataFrame provided."}

    if isinstance(df, dict):
        df = pd.DataFrame.from_dict(df)

    df_copy = df.replace('', np.nan)
    initial_cols = set(df_copy.columns)
    threshold = null_threshold_percent / 100.0
    cols_to_drop = []

    for col in df_copy.columns:
        null_ratio = df_copy[col].isnull().sum() / len(df_copy)
        if null_ratio > threshold:
            cols_to_drop.append(col)

    if not cols_to_drop:
        status = f"No columns found with missing values > {null_threshold_percent}%."
        print(f"--- Status: {status} ---")
        return {"dataframe": df, "status": status} # Return original df if no changes
    else:
        df_cleaned = df.drop(columns=cols_to_drop) # Drop from the original df
        dropped_cols_str = ", ".join(cols_to_drop)
        status = f"Dropped {len(cols_to_drop)} columns ({dropped_cols_str}) with > {null_threshold_percent}% missing values."
        print(f"--- Status: {status} ---")
        return {"dataframe": df_cleaned.to_dict(), "status": status}


def drop_all_null_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Drops rows from the DataFrame where all values are missing (NaN or empty string).
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        Dict[str, Any]: A dictionary containing the modified DataFrame under the key 'dataframe'
                        and a status message under the key 'status'.
    """
    print("--- Calling drop_all_null_rows ---")
    if df is None:
         return {"dataframe": None, "status": "Error: No DataFrame provided."}


    if isinstance(df, dict):
        df = pd.DataFrame.from_dict(df)

    initial_rows = len(df)
    df_copy = df.replace('', np.nan)
    df_cleaned_temp = df_copy.dropna(how='all')

    kept_indices = df_cleaned_temp.index
    df_cleaned = df.loc[kept_indices]
    rows_dropped = initial_rows - len(df_cleaned)

    if rows_dropped > 0:
        status = f"Dropped {rows_dropped} rows where all values were missing."
    else:
        status = "No all-missing rows found to drop."

    print(f"--- Status: {status} ---")
    return {"dataframe": df_cleaned.to_dict(), "status": status}
