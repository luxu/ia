from pathlib import Path

from decouple import config
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI
from composio_langchain import ComposioToolSet, Action, App

from composio_folder.tools import Tools

# llm = ChatOpenAI(
#     api_key=config('OPENAI_API_KEY'),
# )

llm = Tools().get_llm()

current_dir = Path(__file__).parent

prompt = hub.pull("hwchase17/openai-functions-agent")

composio_toolset = ComposioToolSet(
    api_key=config('COMPOSIO_API_KEY')
)
tools = composio_toolset.get_tools(actions=['FILETOOL_SEARCH_WORD'])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

task = f"Procurar pela palavra 'google' no diretório {current_dir}\*.txt"
result = agent_executor.invoke({"input": task})
print(result)
