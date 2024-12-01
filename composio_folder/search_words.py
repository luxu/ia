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

prompt = hub.pull("hwchase17/openai-functions-agent")

composio_toolset = ComposioToolSet(
    api_key=config('COMPOSIO_API_KEY')
)
tools = composio_toolset.get_tools(actions=['FILETOOL_LIST_FILES'])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

task = "Listar arquivos da pasta E:\ia"
result = agent_executor.invoke({"input": task})
print(result)
