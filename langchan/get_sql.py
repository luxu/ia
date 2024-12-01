import os
from decouple import config

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain_openai
from langchain_groq import ChatGroq

# groq_api_key = config('OPENAI_API_KEY')
groq_api_key = config('GROQ_API_KEY')

# model = ChatGroq(model='llama-3.1-70b-versatile')
model = ChatGroq(
    model='llama-3.1-70b-versatile',
    api_key=groq_api_key,
)

db = SQLDatabase.from_uri('sqlite:///db3.sqlite')

toolkit = SQLDatabaseToolkit(
    db=db,
    ll=model,
)
system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
Use as ferramentas necessárias para responder perguntas relacionadas ao histórico de IPCA AO LONGO DOS ANOS
Responda tudo em português brasileiro.
Perguntas: {q}
'''

prompt_template = PromptTemplate.from_template(prompt)

question = '''
Baseado nos dados históricos de IPCA de 2004,
faça uma previsão dos valores de IPCA de cada mês futuro até o final de 2024.
'''

output = agent_executor.invoke({
    'input': prompt_template.format(q=question),
})

print(output.get('output'))
