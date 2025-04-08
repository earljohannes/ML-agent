from pydantic import BaseModel, Field # Use v1 for compatibility if needed
from typing import Literal, Optional, List, Type
from langchain_core.tools import BaseTool
import pandas as pd

from langchain_core.tools import BaseTool


from ml_agent.data_cleaning_agent.toolkit.drop import (
    get_null_value_counts,
    drop_high_null_columns,
    drop_all_null_rows,
)

from ml_agent.data_cleaning_agent.toolkit.imputation import (
    impute_with_simple_strategy,
    impute_with_knn,
    impute_with_bayesian_ridge,
    impute_with_conditional_mode,
)



class MissingValueSchema(BaseModel):
    ...


class ListColumnSchema(BaseModel):
    ...


class DropColsSchema(BaseModel):
    null_threshold_percent: float = Field(..., description="The percent of missing values above which column should be dropped.")


class DropRowsSchema(BaseModel):
    ...


class SimpleImputeSchema(BaseModel):
    column_name: str = Field(..., description="The name of the column to impute.")
    strategy: Literal['mean', 'median', 'most_frequent'] = Field(..., description="The imputation strategy to use ('mean', 'median', or 'most_frequent'). 'mean'/'median' require numeric columns.")


class KNNImputeSchema(BaseModel):
    column_name: str = Field(..., description="The name of the column to impute (target column).")
    n_neighbors: int = Field(default=5, description="Number of neighbors to use for imputation.")
    feature_columns: Optional[List[str]] = Field(default=None, description="Optional list of column names to use as features. If None, uses other numeric columns. Non-numeric features might cause errors or warnings.")


class BayesianRidgeImputeSchema(BaseModel):
    column_name: str = Field(..., description="The name of the numeric column to impute.")
    feature_columns: Optional[List[str]] = Field(default=None, description="Optional list of *numeric* column names to use as features. If None, uses other numeric columns.")


class ConditionalModeImputeSchema(BaseModel):
    target_column: str = Field(..., description="The name of the column where missing values should be imputed.")
    known_column: str = Field(..., description="The name of the column whose values determine the imputation mode.")


class ListColumnsTool(BaseTool):
    name: str = "list_columns"
    description: str = (
        "Get the list of all columns in a dataframe"
    )
    args_schema: Type[BaseModel] = ListColumnSchema
    # This tool needs the DataFrame provided by the execution context
    df_state: dict # Placeholder to indicate need

    def _run(self) -> str:
        """Executes the simple imputation and returns status."""
        # In a real agent, self.df_state would be populated by the calling node
        if not hasattr(self, 'df_state') or self.df_state is None:
             return "Error: DataFrame state not provided to DropRowsTool."

        df = pd.DataFrame.from_dict(self.df_state)
        df_copy: pd.DataFrame = df.copy()
        df_copy.to_csv("./data_clean.csv")

        result = df.columns
        return result


class MissingValuesPercentTool(BaseTool):
    name: str = "missing_value_percent"
    description: str = (
        "Report the percentage of missing values in the dataframe, for each column"
    )
    args_schema: Type[BaseModel] = MissingValueSchema
    df_state: dict # Placeholder to indicate need

    def _run(self) -> str:
        """Returns the null values percentage."""

        df = pd.DataFrame.from_dict(self.df_state)
        return get_null_value_counts(df)


class DropRowsTool(BaseTool):
    name: str = "row_dropper"
    description: str = (
        "Deletes a row if all its attributes are "
    )
    args_schema: Type[BaseModel] = DropRowsSchema
    # This tool needs the DataFrame provided by the execution context
    df_state: dict # Placeholder to indicate need

    def _run(self) -> str:
        """Executes the simple imputation and returns status."""
        # In a real agent, self.df_state would be populated by the calling node
        if not hasattr(self, 'df_state') or self.df_state is None:
             return "Error: DataFrame state not provided to DropRowsTool."

        df = pd.DataFrame.from_dict(self.df_state)
        result = drop_all_null_rows(df)

        self.df_state = result["dataframe"]
        df.to_csv("./data_clean.csv")
        return result["status"]


class DropColsTool(BaseTool):
    name: str = "column_dropper"
    description: str = (
        "Drop a column from dataframe if it has null values higher than specified threshold."
    )
    args_schema: Type[BaseModel] = DropColsSchema
    # This tool needs the DataFrame provided by the execution context
    df_state: dict # Placeholder to indicate need

    def _run(self, null_threshold_percent: float = 50.0) -> str:
        """Executes the simple imputation and returns status."""
        # In a real agent, self.df_state would be populated by the calling node
        if not hasattr(self, 'df_state') or self.df_state is None:
             return "Error: DataFrame state not provided to DropColsTool."

        df = pd.DataFrame.from_dict(self.df_state)
        result = drop_high_null_columns(df, null_threshold_percent=null_threshold_percent)

        self.df_state = result["dataframe"]
        cleaner_df = pd.DataFrame.from_dict(self.df_state)
        cleaner_df.to_csv("./data_clean.csv")

        
        print(f"Tool call complete: {self.name}")
        print(cleaner_df.info())
        return result["status"]


class SimpleImputeTool(BaseTool):
    name: str = "simple_imputer"
    description: str = (
        "Imputes missing values (NaN) in a single column using a simple strategy: "
        "'mean' (numeric only), 'median' (numeric only), or 'most_frequent' (mode, for any type). "
        "Returns a status message indicating success or failure."
    )
    args_schema: Type[BaseModel] = SimpleImputeSchema
    # This tool needs the DataFrame provided by the execution context
    df_state: dict # Placeholder to indicate need

    def _run(self, column_name: str, strategy: Literal['mean', 'median', 'most_frequent']) -> str:
        """Executes the simple imputation and returns status."""
        # In a real agent, self.df_state would be populated by the calling node
        if not hasattr(self, 'df_state') or self.df_state is None:
             return "Error: DataFrame state not provided to SimpleImputeTool."

        df = pd.DataFrame.from_dict(self.df_state)
        result = impute_with_simple_strategy(df, column_name=column_name, strategy=strategy)
        
        df[column_name] = result["series"]
        self.df_state = df

        cleaner_df = pd.DataFrame.from_dict(self.df_state)
        cleaner_df.to_csv("./data_clean.csv")
        print(f"Tool call complete: {self.name}")
        print(cleaner_df.info())

        return result["status"]


class KNNImputeTool(BaseTool):
    name: str = "knn_imputer"
    description: str = (
        "Imputes missing values (NaN) in a target column using K-Nearest Neighbors based on other feature columns. "
        "Works best with numeric features; non-numeric features may cause warnings or errors. "
        "Specify the target column, optionally the number of neighbors (default 5), and optionally a list of feature columns "
        "(defaults to other numeric columns if not specified). Returns a status message."
    )
    args_schema: Type[BaseModel] = KNNImputeSchema
    df_state: dict # Placeholder

    def _run(self, column_name: str, n_neighbors: int = 5, feature_columns: Optional[List[str]] = None) -> str:
        """Executes KNN imputation and returns status."""
        if not hasattr(self, 'df_state') or self.df_state is None:
             return "Error: DataFrame state not provided to KNNImputeTool."

        df = pd.DataFrame.from_dict(self.df_state)
        result = impute_with_knn(df, column_name=column_name, n_neighbors=n_neighbors, feature_columns=feature_columns)

        df[column_name] = result["series"]
        self.df_state = df
        cleaner_df = pd.DataFrame.from_dict(self.df_state)
        cleaner_df.to_csv("./data_clean.csv")
        print(f"Tool call complete: {self.name}")
        print(cleaner_df.info())

        return result["status"]

class BayesianRidgeImputeTool(BaseTool):
    name: str = "bayesian_ridge_imputer"
    description: str = (
        "Imputes missing values (NaN) in a *numeric* target column using Bayesian Ridge regression. "
        "Predicts missing values based on other *numeric* feature columns. "
        "Specify the target numeric column and optionally a list of numeric feature columns "
        "(defaults to other numeric columns if not specified). Returns a status message."
    )
    args_schema: Type[BaseModel] = BayesianRidgeImputeSchema
    df_state: dict # Placeholder

    def _run(self, column_name: str, feature_columns: Optional[List[str]] = None) -> str:
        """Executes Bayesian Ridge imputation and returns status."""
        if not hasattr(self, 'df_state') or self.df_state is None:
             return "Error: DataFrame state not provided to BayesianRidgeImputeTool."

        df = pd.DataFrame.from_dict(self.df_state)
        result = impute_with_bayesian_ridge(df, column_name=column_name, feature_columns=feature_columns)

        df[column_name] = result["series"]
        self.df_state = df
        # Calling code updates df_state using result['series']
        df.to_csv("./data_clean.csv")
        return result["status"]

class ConditionalModeImputeTool(BaseTool):
    name: str = "conditional_mode_imputer"
    description: str = (
        "Imputes missing values (NaN) in a target column (typically categorical) based on the mode (most frequent value) "
        "of that column, conditioned on the values in another specified 'known_column'. "
        "For rows where the conditional mode cannot be determined or the known_column value is missing, "
        "it falls back to the global mode of the target column. Specify the target column and the known column. "
        "Returns a status message."
    )
    args_schema: Type[BaseModel] = ConditionalModeImputeSchema
    df_state: dict # Placeholder

    def _run(self, target_column: str, known_column: str) -> str:
        """Executes conditional mode imputation and returns status."""
        if not hasattr(self, 'df_state') or self.df_state is None:
             return "Error: DataFrame state not provided to ConditionalModeImputeTool."

        df = pd.DataFrame.from_dict(self.df_state)
        result = impute_with_conditional_mode(df, target_column=target_column, known_column=known_column)

        df[target_column] = result["series"]
        self.df_state = df
        df.to_csv("./data_clean.csv")
        return result["status"]
