from typing import Annotated, List, Sequence, TypedDict
from operator import add
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.pregel import RunnableConfig
from langchain_core.tools import BaseTool
from ml_agent.data_cleaning_agent.toolkit.toolkit import ImputationToolkit


system_message = """
You are an agent specialized to clean a pandas Dataframe.

You have access to tools for cleaning the dataframe.

Only use the below tools. Only use the information returned by the below tools to clean the data.

To start you should ALWAYS look at the columns in the dataframe.
Do NOT skip this step.

Then, for each of the columns, you should select appropriate tools to clean the column according to data type and distribution of that column.

If there are no missing values in the dataframe, do not use any tool.
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    query: str
    question: str
    system_message: str


class DataCleaningAgent:
    """Agent for data cleaning."""

    def __init__(self, llm, tools: Sequence[BaseTool]):
        self.llm = llm
        self.imputation_tools = tools


    def _create_agent_workflow(self) -> StateGraph:
        """Creates the agent workflow."""

        def add_system_prompt(state: AgentState):
            if not ("system_message" in state and state.get("system_message")):
                state["system_message"] = system_message
                state["messages"].append(SystemMessage(system_message))
                
            result = self.llm.invoke(system_message)
            return {"messages": [result]}


        def call_model(state: AgentState, config: RunnableConfig):
            """Invokes the LLM to decide the next step or generate a response."""
            print("🔥🔥 Calling Model 🔥🔥", state["question"])
            state["messages"].append(HumanMessage(state['question']))

            model_with_tools = self.llm.bind_tools(self.imputation_tools)

            response: AIMessage = model_with_tools.invoke(state["messages"], config=config)
            print(f"🎯 Model Response: {response.content} ")
            print(f"🎯 Model Tool Calls: {response.tool_calls} ")

            # The response from the LLM is appended to the 'messages' list in the state
            return {"messages": [response]}


        def call_tool(state: AgentState):
            """Executes the tool called by the LLM."""
            print("🔥🔥 Calling Tool 🔥🔥")
            last_message: AIMessage = state['messages'][-1] # Get the latest AI message

            if not last_message.tool_calls:
                print("🪲 ERROR: No tool calls found in the last message")
                return {"messages": []} # No new messages if no tool call was actually made

            tool_messages: List[ToolMessage] = []

            # support multiple tool calls in one turn if needed
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                print(f"--- Executing Tool: {tool_name} ---")

                # Find the corresponding tool function
                selected_tool = None
                for t in self.imputation_tools:
                    if t.name == tool_name:
                        selected_tool = t
                        break

                if selected_tool:
                    try:
                        # Execute the tool with the arguments provided by the LLM
                        tool_output = selected_tool.invoke(tool_call["args"])
                        print(f"--- Tool Output: {tool_output} ---")

                        tool_messages.append(
                            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                        )
                    except Exception as e:
                        print(f"--- ERROR executing tool {tool_name}: {e} ---")

                        tool_messages.append(
                            ToolMessage(content=f"Error executing tool {tool_name}: {str(e)}", tool_call_id=tool_call["id"])
                        )
                else:
                    print(f"--- ERROR: Tool '{tool_name}' not found ---")
                    tool_messages.append(
                        ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call["id"])
                    )

            return {"messages": tool_messages}


        def should_continue(state: AgentState) -> str:
            """Determines whether to continue the loop (call a tool) or end."""
            print("Checking Condition: Should Continue 🤔? ")
            last_message = state['messages'][-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                print("Decision: Continue ✅(Call Tool) ---")
                return "call_tool"

            else:
                print("Decision: End 🛑")
                return END
        

        workflow = StateGraph(AgentState)

        workflow.add_node("add_system_prompt", add_system_prompt)
        workflow.add_node("call_model", call_model)
        workflow.add_node("action", call_tool)

        workflow.add_edge(START, "add_system_prompt")
        workflow.add_edge("add_system_prompt", "call_model")
        workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "call_tool": "action",
                END: END
            }
        )
        workflow.add_edge("action", "call_model")
        app = workflow.compile()

        return app
