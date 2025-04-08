from typing import List, Dict, Any
from langchain_core.tools import BaseTool, BaseToolkit


from ml_agent.data_cleaning_agent.toolkit.tools import (
    ListColumnsTool,
    MissingValuesPercentTool,
    DropRowsTool,
    DropColsTool,
    SimpleImputeTool,
    KNNImputeTool,
    BayesianRidgeImputeTool,
    ConditionalModeImputeTool,
)


class ImputationToolkit(BaseToolkit):
    """Toolkit containing tools for imputing missing values in a DataFrame."""
    # The toolkit itself doesn't hold the df_state, the individual tools do when instantiated by the agent
    df_state: dict # Indicates the toolkit operates in a context with a DataFrame

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        # Pass the DataFrame state to each tool instance
        if not hasattr(self, 'df_state') or self.df_state is None:
             raise ValueError("DataFrame state must be set in ImputationToolkit before getting tools.")

        return [
            ListColumnsTool(df_state=self.df_state),
            MissingValuesPercentTool(df_state=self.df_state),
            DropColsTool(df_state=self.df_state),
            DropRowsTool(df_state=self.df_state),
            SimpleImputeTool(df_state=self.df_state),
            KNNImputeTool(df_state=self.df_state),
            BayesianRidgeImputeTool(df_state=self.df_state),
            ConditionalModeImputeTool(df_state=self.df_state),
        ]
