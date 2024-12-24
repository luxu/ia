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
        Assinale a alternativa correta. Quais são os cinco princípios da Direção Defensiva:
        Escolha uma opção:
        a. Conhecimento, Atenção, Preferência, Habilidade, e Ação.
        b. Conhecimento, Aferição, Prestação de socorro, Auxilio mecânico e direção.
        c. Conhecimento, Atenção, Previsão, Habilidade, e Ação.
        d. Conhecimento, Atenção, Previsão, Habilitação, e Ação.
         """
     },
    {"Pergunta1":
         """
        De acordo com o estudado neste bloco, Direção Defensiva é o:
        Escolha uma opção:
        a. Conjunto de ações e posturas tomadas para se conduzir um veículo no trânsito, 
        tornando possível reconhecer antecipadamente as condições adversas, possibilitando prevenir ou minimizar a 
        ocorrência de acidentes.
        b. Conjunto de ações e posturas tomadas para se conduzir um veículo no trânsito, ao qual impossibilita o 
        condutor de reconhecer antecipadamente as condições adversas, impossibilitando prevenir ou minimizar a 
        ocorrência de acidentes.
        c. Conjunto de ações e posturas não tomadas na condução de um veículo no trânsito, tornando possível 
        evitar a ocorrência de acidentes.
        d. Todas alternativas estão incorretas."""
     },
    {"Pergunta2":
         """
        Neste bloco de estudo sobre Direção defensiva, abordamos as Condições adversas no trânsito,  e apresentamos seis fatores de risco, quais são? Assinale a alternativa correta:
        Escolha uma opção:
        a. Nenhuma das alternativas estão corretas.
        b. Iluminação, trânsito, veículo, ferrovia, morro, condutor.
        c. Iluminação, pontilhão, veículo, tempo, via, passarela.
        d. Iluminação, trânsito, veículo, tempo, via, condutor.
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
