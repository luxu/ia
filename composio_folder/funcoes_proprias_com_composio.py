from langchain_core.tools import tool

from composio_folder.tools import Tools

CUSTOMERS = [
    {'nome': 'Felipe', 'idade': 27},
    {'nome': 'Jonas', 'idade': 18},
    {'nome': 'Ana', 'idade': 30},
    {'nome': 'Ronaldo', 'idade': 40},
    {'nome': 'Gabriele', 'idade': 26},
]


@tool
def get_customers(nome):
    '''
    Retorna todos os clientes da base de dados.
    '''
    return CUSTOMERS


@tool
def get_customer_by_name(name: str):
    '''
    Busca e retorna um cliente pelo nome.
    Args:
        name: Nome do cliente
    '''
    for customer in CUSTOMERS:
        if customer['nome'].lower() == name.lower():
            return customer
    return None

words_to_system = 'Você é um assistente e deve usar suas ferramentas para responder.'
tools = [get_customers, get_customer_by_name]
question = input('Como posso ajudar? ')
tools = Tools(tools, question, words_to_system)
print(tools.get_result())
