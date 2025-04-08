from typing import List, Optional, TypedDict, Union
from typing_extensions import Annotated
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from operator import add
from ml_agent.feature_transform_agent.toolkit.toolkit import FeatureEngineeringToolkit


class FeatureEngineeringState(TypedDict):
    messages: Annotated[List[BaseMessage], add] # Accumulates messages
    dataframe: pd.DataFrame              # The DataFrame being modified
    dataframe_info: str                  # String representation of df.info() for context
    original_request: str                # User's initial request (for context)
    action_log: List[str]                # Log of successful actions
    error: Optional[str]                 # Any error message encountered


def get_df_info(df: pd.DataFrame) -> str:
    """Gets DataFrame info as a string."""
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()
