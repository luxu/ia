from composio_langchain import ComposioToolSet, App
from decouple import config

from composio_folder.tools import Tools

composio_toolset = ComposioToolSet(
    api_key=config('COMPOSIO_API_KEY'),
)

tools = composio_toolset.get_tools(
    apps=[
        App.GOOGLESHEETS,
        # App.GMAIL,
    ],
)

words_to_system = 'Você é um assistente responsável por fazer a gestão de planilhas.'
question = input('Como posso ajudar? ')
tools = Tools(tools, question, words_to_system)
print(tools.get_result())
