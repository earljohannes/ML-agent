from langchain_core.tools import BaseTool
import pandas as pd
import numpy as np
from typing import List, Optional, Literal, Union, Dict, Any, Tuple, Type, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from operator import add
from ml_agent.feature_transform_agent.toolkit.transform import implement_scale_features, implement_to_datetime
from ml_agent.feature_transform_agent.toolkit.encoding import implement_one_hot_encode, implement_label_encode, implement_target_encode
from sklearn.preprocessing import TargetEncoder
from abc import ABC


class ScaleFeaturesInput(BaseModel):
    column: str = Field(..., description="The numerical column name to scale.")
    method: Literal['standard', 'minmax'] = Field(..., description="The scaling method ('standard' or 'minmax').")
    output_column_suffix: str = Field("_scaled", description="Suffix to append to the original column name for the new scaled column.")

class OneHotEncodeInput(BaseModel):
    column: str = Field(..., description="The categorical/object column name to one-hot encode.")
    drop_original: bool = Field(True, description="Whether to drop the original column after encoding.")
    dummy_na: bool = Field(False, description="Whether to add a column to indicate NaNs, if False NaNs are ignored.")

class LabelEncodeInput(BaseModel):
    column: str = Field(..., description="The categorical/object column name to label encode. This replaces the original column.")

class TargetEncodeInput(BaseModel):
    column_to_encode: str = Field(..., description="The categorical column name to target encode.")
    target_column: str = Field(..., description="The numerical target column name used for encoding.")
    output_column_suffix: str = Field("_target_encoded", description="Suffix to append to the original column name for the new target encoded column.")
    drop_original: bool = Field(True, description="Whether to drop the original column after encoding.")

class ToDatetimeInput(BaseModel):
    column: str = Field(..., description="The column name (string or object type) to convert to datetime objects.")
    datetime_format: Optional[str] = Field(None, description="The specific format string of the date/time if known (e.g., '%Y-%m-%d %H:%M:%S'). If None, pandas will try to infer.")
    coerce_errors: bool = Field(True, description="If True, parsing errors will result in NaT (Not a Time). If False, raise an error.")


class ScaleFeaturesTool(BaseTool):
    name: str = "scale_numerical_features"
    description: str = "Applies Standard (Z-score) or Min-Max scaling to a numerical column, creating a new column with the scaled values."
    args_schema: Type[BaseModel] = ScaleFeaturesInput

    def _run(self, column: str, method: Literal['standard', 'minmax'], output_column_suffix: str, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """ Tool execution logic """
        # Note: current_df is passed explicitly by the execution node, not part of args_schema
        return implement_scale_features(current_df, column, method, output_column_suffix)


class OneHotEncodeTool(BaseTool):
    name: str = "one_hot_encode_categorical_feature"
    description: str = "Converts a categorical column into multiple binary (0/1) columns, one for each category. Optionally drops the original column."
    args_schema: Type[BaseModel] = OneHotEncodeInput

    def _run(self, column: str, drop_original: bool, dummy_na: bool, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """ Tool execution logic """
        return implement_one_hot_encode(current_df, column, drop_original, dummy_na)


class LabelEncodeTool(BaseTool):
    name: str = "label_encode_categorical_feature"
    description: str = "Converts a categorical column into numerical labels (0, 1, 2,...). Replaces the original column."
    args_schema: Type[BaseModel] = LabelEncodeInput

    def _run(self, column: str, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """ Tool execution logic """
        return implement_label_encode(current_df, column)


class TargetEncodeTool(BaseTool):
    name: str = "target_encode_categorical_feature"
    description: str = "Encodes a categorical column based on the mean of a specified numerical target variable for each category. Requires 'category_encoders' library."
    args_schema: Type[BaseModel] = TargetEncodeInput

    def _run(self, column_to_encode: str, target_column: str, output_column_suffix: str, drop_original: bool, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """ Tool execution logic """
        if TargetEncoder is None:
             raise RuntimeError("TargetEncoder tool requires the 'category_encoders' library, which is not installed.")
        return implement_target_encode(current_df, column_to_encode, target_column, output_column_suffix, drop_original)


class ToDatetimeTool(BaseTool):
    name: str = "convert_to_datetime"
    description: str = "Converts a column (usually string or object type) containing date/time information into datetime objects. Replaces the original column."
    args_schema: Type[BaseModel] = ToDatetimeInput

    def _run(self, column: str, datetime_format: Optional[str], coerce_errors: bool, current_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """ Tool execution logic """
        return implement_to_datetime(current_df, column, datetime_format, coerce_errors)



