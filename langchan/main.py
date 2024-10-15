import os

from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_experimental.utilities import PythonREPL

os.environ["GOOGLE_API_KEY"] = config("API_GEMINI")

# ChatGoogleGenerativeAI(model="gemini-pro-vision")

# model = ChatOpenAI(model='gpt-3.5-turbo')
model = ChatGoogleGenerativeAI(model="gemini-pro")

response = model.invoke("Como sugerir e rodar um comando Python numa LLM?")

print(response.content)


# model = ChatOpenAI(model='gemini-pro')

# model = ChatOpenAI(model='gpt-4')
# name_bd = 'db.sqlite3'

# db = SQLDatabaseToolkit.from_uri(f'sqlite:///{name_bd}')
# db = SQLDatabase.from_uri(f'sqlite:///{name_bd}')

# toolkit = SQLDatabaseToolkit(
#     db=db,
#     llm=model
# )
#
# system_message = hub.pull('hwchase17/react')
#
# agent = create_react_agent(
#     llm=model,
#     tools=toolkit.get_tools(),
#     prompt=system_message
# )
#
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=toolkit.get_tools(),
#     verbose=True,
# )


prompt = """
Como assistente pessoal que responderá as perguntas dando dicas de gastar menos no cartão.
Responda tudo em português brasileiro.
Perguntas: {q}
"""

# prompt_template = PromptTemplate.from_template(prompt)
#
# question = 'Qual mês e ano tiveram o maior gasto?'
#
# output = agent_executor.invoke({
#     'input': prompt_template.format(q=question),
# })
#
# print(output.get('output'))

# python_repl = PythonREPL()
# python_repl_tool = Tool(
#     name='Python REPL',
#     description='Um shell Python. Use isso para executar código Python. Execute apenas códigos Python válidos.'
#                 'Se você precisar obter o retorno do código, use a função "print(...)"'
#                 'Use para realizar cálculos financeiros necessários para responder as perguntas e dar dicas.',
#     func=python_repl.run,
# )
#
# print(python_repl_tool)
