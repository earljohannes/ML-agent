from typing import List
from abc import ABC
from sklearn.preprocessing import TargetEncoder
from ml_agent.feature_transform_agent.toolkit.tools import ScaleFeaturesTool, OneHotEncodeTool, LabelEncodeTool, TargetEncodeTool, ToDatetimeTool
from langchain_core.tools import BaseToolkit, BaseTool


class FeatureEngineeringToolkit(BaseToolkit, ABC):
    """Toolkit containing tools for feature scaling, encoding, and type transformation."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools = [
            ScaleFeaturesTool(),
            OneHotEncodeTool(),
            LabelEncodeTool(),
            ToDatetimeTool(),
        ]

        if TargetEncoder is not None: # Only include if library is available
            tools.append(TargetEncodeTool())

        return tools
