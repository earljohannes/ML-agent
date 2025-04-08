import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from typing import Literal, Optional, List, Type, Dict, Any


def impute_with_simple_strategy(df: pd.DataFrame, column_name: str, strategy: str = 'mean') -> Dict[str, Any]:
    """Imputes using mean, median, or mode. Returns status and imputed series."""
    result = {"status": "", "series": None, "error": None}
    if column_name not in df.columns:
        result["error"] = f"Column '{column_name}' not found."
        result["status"] = f"Error: {result['error']}"
        return result

    data_column = df[[column_name]].copy()
    original_series = df[column_name] # Keep original for return on error

    if strategy in ['mean', 'median'] and not pd.api.types.is_numeric_dtype(data_column[column_name]):
        result["error"] = f"Strategy '{strategy}' requires numeric data, but column '{column_name}' is not numeric."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result

    try:
        imputer = SimpleImputer(strategy=strategy)
        imputed_data = imputer.fit_transform(data_column)
        imputed_series = pd.Series(imputed_data.flatten(), index=df.index, name=column_name)
        result["series"] = imputed_series
        result["status"] = f"Successfully imputed column '{column_name}' using strategy '{strategy}'."
        print(result["status"]) # Keep console log
    except Exception as e:
        result["error"] = f"Error during '{strategy}' imputation for column '{column_name}': {e}"
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series # Return original on error
        print(result["status"])

    return result


def impute_with_knn(df: pd.DataFrame, column_name: str, n_neighbors: int = 5, feature_columns: list = None) -> Dict[str, Any]:
    """Imputes using KNN. Returns status and imputed series."""
    result = {"status": "", "series": None, "error": None}
    original_series = df.get(column_name, None)
    if original_series is None:
        result["error"] = f"Target column '{column_name}' not found."
        result["status"] = f"Error: {result['error']}"
        return result

    # Input validation and feature selection (as before)
    if not pd.api.types.is_numeric_dtype(df[column_name]):
         print(f"Warning: KNNImputer works best with numeric target column ('{column_name}').") # Keep as warning

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=np.number).columns.tolist()
        if column_name in feature_columns: feature_columns.remove(column_name)
    else:
        valid_features = []
        numeric_feature_found = False
        for col in feature_columns:
            if col == column_name: continue
            if col not in df.columns: print(f"Warning: Feature '{col}' not found. Skipping."); continue
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_features.append(col)
                numeric_feature_found = True
            else:
                 print(f"Warning: Feature '{col}' is not numeric. KNN may fail. Consider encoding.")
                 # Decide whether to include non-numeric or not. Let's include for now.
                 valid_features.append(col)
        feature_columns = valid_features

    if not feature_columns:
        result["error"] = f"No suitable feature columns found/specified for KNN imputation of '{column_name}'."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result
    if not numeric_feature_found:
         print(f"Warning: No strictly numeric feature columns provided/found for KNN imputation of '{column_name}'. KNN may fail.")


    cols_for_imputation = [column_name] + feature_columns
    data_subset = df[cols_for_imputation].copy()

    try:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(data_subset)
        imputed_df = pd.DataFrame(imputed_data, columns=cols_for_imputation, index=df.index)
        result["series"] = imputed_df[column_name]
        result["status"] = f"Successfully imputed column '{column_name}' using KNN (k={n_neighbors}) with features: {feature_columns}."
        print(result["status"])
    except Exception as e:
        result["error"] = f"Error during KNN imputation for '{column_name}': {e}"
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        print(result["status"])

    return result


def impute_with_bayesian_ridge(df: pd.DataFrame, column_name: str, feature_columns: list = None) -> Dict[str, Any]:
    """Imputes using Bayesian Ridge. Returns status and imputed series."""
    result = {"status": "", "series": None, "error": None}
    original_series = df.get(column_name, None)
    if original_series is None:
        result["error"] = f"Target column '{column_name}' not found."
        result["status"] = f"Error: {result['error']}"
        return result

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        result["error"] = f"Bayesian Ridge requires a numeric target column ('{column_name}')."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result

    # Feature selection/validation (only numeric allowed for Bayesian Ridge)
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=np.number).columns.tolist()
        if column_name in feature_columns: feature_columns.remove(column_name)
    else:
        valid_features = []
        for col in feature_columns:
            if col == column_name: continue
            if col not in df.columns: print(f"Warning: Feature '{col}' not found. Skipping."); continue
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_features.append(col)
            else:
                 print(f"Warning: Feature '{col}' is not numeric. Skipping for Bayesian Ridge.")
        feature_columns = valid_features

    if not feature_columns:
        result["error"] = f"No suitable *numeric* feature columns found/specified for Bayesian Ridge imputation of '{column_name}'."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result

    # Prepare data and handle NaNs in features (as before)
    df_copy = df[[column_name] + feature_columns].copy()
    missing_mask = df_copy[column_name].isnull()
    X_train = df_copy.loc[~missing_mask, feature_columns]
    y_train = df_copy.loc[~missing_mask, column_name]
    X_predict = df_copy.loc[missing_mask, feature_columns]

    if X_train.empty or y_train.empty:
        result["error"] = f"No non-missing values in '{column_name}' to train Bayesian Ridge model."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result
    if X_predict.empty:
        result["status"] = f"Info: No missing values found in '{column_name}' to impute with Bayesian Ridge."
        result["series"] = original_series # Return original as no change needed
        return result

    feature_imputer = SimpleImputer(strategy='mean')
    try:
        X_train_imputed = feature_imputer.fit_transform(X_train)
        X_predict_imputed = feature_imputer.transform(X_predict)
    except ValueError as e:
        result["error"] = f"Error imputing feature columns for Bayesian Ridge model: {e}. Ensure features are numeric."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result

    try:
        model = BayesianRidge()
        model.fit(X_train_imputed, y_train)
        predicted_values = model.predict(X_predict_imputed)

        imputed_series = df[column_name].copy()
        imputed_series.loc[missing_mask] = predicted_values
        result["series"] = imputed_series
        result["status"] = f"Successfully imputed column '{column_name}' using Bayesian Ridge Regression with features: {feature_columns}."
        print(result["status"])
    except Exception as e:
        result["error"] = f"Error during Bayesian Ridge imputation for '{column_name}': {e}"
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series # Return original on error
        print(result["status"])

    return result


def impute_with_conditional_mode(df: pd.DataFrame, target_column: str, known_column: str) -> Dict[str, Any]:
    """Imputes using conditional mode. Returns status and imputed series."""
    result = {"status": "", "series": None, "error": None}
    original_series = df.get(target_column, None)
    if original_series is None or known_column not in df.columns :
        result["error"] = f"One or both columns ('{target_column}', '{known_column}') not found."
        result["status"] = f"Error: {result['error']}"
        return result

    if pd.api.types.is_numeric_dtype(df[target_column]):
         print(f"Warning: Target '{target_column}' is numeric. Conditional mode usually for categorical.")

    imputed_series = df[target_column].copy()
    missing_mask = imputed_series.isnull()

    if not missing_mask.any():
        result["status"] = f"Info: No missing values found in '{target_column}'."
        result["series"] = imputed_series
        return result

    # Calculate conditional modes and global fallback (as before)
    try:
        conditional_modes = df[~missing_mask].groupby(known_column)[target_column].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        conditional_modes_map = conditional_modes.to_dict()
    except Exception as e:
        result["error"] = f"Error calculating conditional modes: {e}."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result

    global_mode = df[target_column].mode()
    fallback_value = global_mode[0] if not global_mode.empty else np.nan

    # Apply imputation (as before)
    imputed_count = 0
    fallback_count = 0
    nan_remain_count = 0
    for index in imputed_series[missing_mask].index:
        known_value = df.loc[index, known_column]
        imputed_value = conditional_modes_map.get(known_value) if pd.notna(known_value) else None

        if pd.notna(imputed_value):
             imputed_series.loc[index] = imputed_value
             imputed_count += 1
        elif pd.notna(fallback_value):
             imputed_series.loc[index] = fallback_value
             fallback_count += 1
        else:
             nan_remain_count += 1


    result["series"] = imputed_series
    status_parts = [f"Successfully imputed column '{target_column}' using conditional mode based on '{known_column}'."]
    if imputed_count > 0: status_parts.append(f"{imputed_count} values imputed via conditional mode.")
    if fallback_count > 0: status_parts.append(f"{fallback_count} values imputed via global mode ('{fallback_value}').")
    if nan_remain_count > 0: status_parts.append(f"{nan_remain_count} values remain NaN (no conditional or global mode found).")
    result["status"] = " ".join(status_parts)
    print(result["status"])

    return result


def impute_with_interpolation(df: pd.DataFrame, column_name: str, method: str = 'linear') -> Dict[str, Any]:
    """Imputes using interpolation. Returns status and imputed series."""
    result = {"status": "", "series": None, "error": None}
    original_series = df.get(column_name, None)
    if original_series is None:
        result["error"] = f"Column '{column_name}' not found."
        result["status"] = f"Error: {result['error']}"
        return result

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        result["error"] = f"Interpolation requires numeric data, but column '{column_name}' is not numeric."
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series
        return result

    try:
        imputed_series = df[column_name].interpolate(method=method)
        result["series"] = imputed_series
        result["status"] = f"Successfully imputed column '{column_name}' using interpolation method '{method}'."
        print(result["status"])
    except Exception as e:
        result["error"] = f"Error during interpolation for column '{column_name}': {e}"
        result["status"] = f"Error: {result['error']}"
        result["series"] = original_series # Return original on error
        print(result["status"])

    return result
