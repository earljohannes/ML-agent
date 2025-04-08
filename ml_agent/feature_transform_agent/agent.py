from typing import Annotated, List, Sequence, Optional, TypedDict, Literal
from operator import add
import pandas as pd
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.pregel import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

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



class FeatureEngineeringAgent:
    def __init__(self, llm, tools: Sequence[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.toolkit = FeatureEngineeringToolkit()
        self.llm_with_tools = self.llm.bind_tools(tools=tools, tool_choice="any")

    def _create_agent_workflow(self) -> StateGraph:

        def start_node(state: FeatureEngineeringState) -> FeatureEngineeringState:
            """Initializes the process and adds the first human message."""
            print("--- Starting Feature Engineering ---")
            df = state['dataframe']
            state['dataframe_info'] = get_df_info(df)
            state['action_log'] = []
            state['error'] = None

            # Add the initial request as the first message
            state['messages'] = [HumanMessage(content=state['original_request'])]

            print("Initial DataFrame Info:")
            print(state['dataframe_info'])
            return state

        def interpret_request_node(state: FeatureEngineeringState) -> FeatureEngineeringState:
            """Interprets the user's request using the LLM with tools."""
            print(f"\n--- Interpreting Request ---")
            current_messages = state['messages']
            original_request = state['original_request'] # Keep original request for context if needed
            df_info = state['dataframe_info']

            # Construct prompt for the LLM
            # Include DataFrame info in the system prompt or first user message for context
            prompt_messages = [
                SystemMessage(content=f"""
        You are an expert feature engineering assistant. Your goal is to understand the user's request
        and select the single most appropriate tool to apply to the pandas DataFrame.

        Current DataFrame Summary:
        {df_info}

        Analyze the latest user message and choose the best tool and parameters to fulfill the request.
        Only select one tool per turn. If the request is ambiguous or cannot be fulfilled with the available tools,
        respond asking for clarification or stating the limitation, but do *not* call a tool.
        """),
                *current_messages # Include history
            ]


            try:
                ai_response = self.llm_with_tools.invoke(prompt_messages)
                print(f"LLM Response: {ai_response.content}") # Log LLM's text response
                # Add AI response (including potential tool calls) to state
                state['messages'] = add(state['messages'], [ai_response])
                state['error'] = None # Clear previous errors if LLM call succeeds

                # Check for tool calls before proceeding
                if not ai_response.tool_calls:
                    print("LLM did not select a tool. It might be asking for clarification or the request is unfulfillable.")
                    # No tool call means we might finish or need more info depending on the response
                    # We'll handle this in the conditional edge

            except Exception as e:
                error_msg = f"Error during LLM call: {e}"
                print(f"Error: {error_msg}")
                state['error'] = error_msg
                # Add error as a message? Or just use the error flag.
                # state['messages'] = add_messages(state['messages'], [SystemMessage(content=f"Error: {error_msg}")])


            return state

        def execute_tool_node(state: FeatureEngineeringState) -> FeatureEngineeringState:
            """Executes the tool chosen by the LLM."""
            print("\n--- Executing Tool ---")
            ai_message = state['messages'][-1] # Get the latest AI message

            if not isinstance(ai_message, AIMessage) or not ai_message.tool_calls:
                print("No tool call found in the last AI message. Skipping execution.")
                # This case should ideally be handled by the conditional edge, but double-check
                return state

            tool_call = ai_message.tool_calls[0] # Assuming one tool call per turn for simplicity
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']

            print(f"Attempting to execute tool: {tool_name} with args: {tool_args}")

            # Find the corresponding tool in the toolkit
            selected_tool : Optional[BaseTool] = None
            for t in self.tools:
                if t.name == tool_name:
                    selected_tool = t
                    break

            if not selected_tool:
                error_msg = f"Error: Tool '{tool_name}' not found in the toolkit."
                print(error_msg)
                state['error'] = error_msg
                state['messages'] = add_messages(state['messages'], [ToolMessage(content=error_msg, tool_call_id=tool_call_id)])
                return state

            # Execute the tool's _run method, passing the current DataFrame explicitly
            current_df = state['dataframe']
            try:
                # Validate arguments using the tool's args_schema before calling _run
                validated_args = selected_tool.args_schema(**tool_args)

                # Call the tool's _run method, passing df and validated args
                # We rely on the specific tool's _run signature accepting current_df
                modified_df, log_message = selected_tool._run(**validated_args.model_dump(), current_df=current_df)

                # Update state on success
                state['dataframe'] = modified_df
                state['action_log'].append(log_message)
                state['dataframe_info'] = get_df_info(modified_df) # Update df info
                state['error'] = None # Clear error on success
                print(f"Execution successful: {log_message}")
                print("Updated DataFrame Info:")
                print(state['dataframe_info'])

                # Add ToolMessage confirming success
                state['messages'] = add_messages(state['messages'], [ToolMessage(content=f"Success: {log_message}", tool_call_id=tool_call_id)])

            except ValidationError as e:
                error_msg = f"Error: Invalid arguments provided for tool '{tool_name}'. Details: {e}"
                print(error_msg)
                state['error'] = error_msg
                state['messages'] = add_messages(state['messages'], [ToolMessage(content=error_msg, tool_call_id=tool_call_id)])

            except (ValueError, TypeError, RuntimeError, KeyError) as e:
                # Catch specific errors from our implementation functions
                error_msg = f"Error executing tool '{tool_name}': {e}"
                print(error_msg)
                state['error'] = error_msg
                state['messages'] = add_messages(state['messages'], [ToolMessage(content=error_msg, tool_call_id=tool_call_id)])
                # Keep the DataFrame as it was before the failed attempt

            except Exception as e:
                # Catch any other unexpected errors
                error_msg = f"Unexpected error executing tool '{tool_name}': {e}"
                print(error_msg)
                state['error'] = error_msg
                state['messages'] = add_messages(state['messages'], [ToolMessage(content=error_msg, tool_call_id=tool_call_id)])

            return state

        def handle_error_node(state: FeatureEngineeringState) -> FeatureEngineeringState:
            """Handles errors encountered during interpretation or execution."""
            print("\n--- Handling Error ---")
            print(f"An error occurred: {state['error']}")
            # Simple handling: just log. The error is already in the state and potentially messages.
            if state['error'] not in state['action_log']: # Avoid duplicate logging if error came from execution
                state['action_log'].append(f"ERROR: {state['error']}")
            # Stop the process
            return state

        def should_execute_or_finish(state: FeatureEngineeringState) -> Literal["execute_tool", "handle_error", END]:
            """Determines the next step after the LLM interprets the request."""
            if state.get('error'):
                print("Condition: Error detected, routing to handle_error")
                return "handle_error"

            last_message = state['messages'][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                print("Condition: LLM selected a tool, routing to execute_tool")
                return "execute_tool"
            else:
                # No tool call, LLM might have responded directly (e.g., asking for clarification or saying done)
                print("Condition: No tool selected or error, routing to END (or potentially a clarification loop)")
                # For this example, we end if no tool is called.
                # Could be extended to loop back to user input or LLM if clarification needed.
                return END

        def after_execution_check(state: FeatureEngineeringState) -> Literal["interpret_request", "handle_error"]:
            """Checks after execution: Continue interpreting (if more steps needed?) or handle error."""
            # This simple agent performs one step per request.
            # If execution had an error, handle it. Otherwise, the single step is done.
            # Let's modify this slightly: After successful execution, we END.
            # If we wanted multi-step execution per request, we'd loop back to 'interpret_request'.

            # Let's refine: The ToolMessage added after execution becomes input for the next LLM turn.
            # So, we should loop back to the LLM to see if it wants to do more based on the result.
            if state.get('error'):
                print("Condition: Execution failed, routing to handle_error")
                return "handle_error"
            else:
                # Decide if more work is needed based on the conversation or a fixed number of steps.
                # For now, let's assume one successful transformation is enough and END.
                # To enable multi-step, change this to return "interpret_request"
                print("Condition: Execution successful, routing back to LLM (interpret_request) for potential next step.")
                # Let's loop back to see if LLM thinks it's done or wants another step.
                return "interpret_request" # Loop back

        workflow = StateGraph(FeatureEngineeringState)

        workflow.add_node("start", start_node)
        workflow.add_node("interpret_request", interpret_request_node)
        workflow.add_node("execute_tool", execute_tool_node)
        workflow.add_node("handle_error", handle_error_node)

        workflow.set_entry_point("start")
        workflow.add_edge("start", "interpret_request")
        workflow.add_conditional_edges(
            "interpret_request",
            should_execute_or_finish,
            {
                "execute_tool": "execute_tool",
                "handle_error": "handle_error",
                END: END,
            }
        )
        workflow.add_conditional_edges(
            "execute_tool",
            after_execution_check,
            {
                "interpret_request": "interpret_request",
                "handle_error": "handle_error",
            }
        )
        workflow.add_edge("handle_error", END)
        app = workflow.compile()
        return app

