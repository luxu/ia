from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from decouple import config

api_key = config("API_GEMINI")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=config('API_GEMINI'),
)

arquivo_py = Path(__file__)

file = Path(r'E:\ia\files\modulo3') / arquivo_py.with_suffix('.pdf').name

if not Path(file).exists():
    print(f"File not found: {file}")
    exit(0)

questions = [
    {"Pergunta0":
         """
        Assinale a alternativa incorreta. Para obter uma condução segura é 
        necessário utilizar algumas técnicas durante a condução:
        Escolha uma opção:
        a. Uso do elemento de segurança – 3º pedal se for carro automático, e 4º pedal se for carro manual.
        b. Use somente uma das palmas das mãos abertas para realizar manobras para direita ou para esquerda, 
        como se estivesse lavando prato.
        c. Uso do freio de estacionamento.
        d. Pegadas no volante – forma correta. Com as mãos correspondentes ao ponteiro de um relógio segure 
        na posição 9h15 ou 10h10.
         """
     },
    {"Pergunta1":
         """
        De acordo com o estudado neste bloco, Ergonomia é a ciência que busca:
        Escolha uma opção:
        a. Todas as alternativas estão incorretas.
        b. Favorecer o veículo para evitar seu desgaste prematuro, visando condicionar o homem a máquina, a fim de 
        melhorar este relacionamento.
        c. Entender a relação do veículo com as condições de seu local de manutenção, estabelecendo normas para 
        melhorar este relacionamento.
        d. Entender a relação do homem com as condições de trabalho, estabelecendo normas para melhorar este 
        relacionamento.
         """
     },
    {"Pergunta2":
         """
        Para obter uma Ergonomia adequada dentro da viatura é necessário fazer os seguintes ajustes:
        Escolha uma opção:
        a. Encosto de tronco, e encosto de cabeça.
        b. Assento.
        c. Volante, cinto de segurança, espelhos retrovisores.
        d. Todas as alternativas estão corretas.
         """
     }
]
choice_questions = questions[2]

loader = PyPDFLoader(file)
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Seja um aluno da EAP, 
        respondendo perguntas sobre o 
        o conteúdo ministrado em aula.
        Contexto: {context}
        Pergunta: {question}"""
)

chain = prompt_base_conhecimento | model | StrOutputParser()

response = chain.invoke(
    {
        "context": '\n'.join(doc.page_content for doc in documents),
        "question": choice_questions,
    }
)

print(response)
