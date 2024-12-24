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

file = Path(r'E:\ia\files') / arquivo_py.with_suffix('.pdf').name

if not Path(file).exists():
    print(f"File not found: {file}")
    exit(0)

loader = PyPDFLoader(file)
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Seja um aluno da EAP, respondendo perguntas sobre o conteúdo ministrado em aula.
        Contexto: {context}
        Pergunta: {question}"""
)

chain = prompt_base_conhecimento | model | StrOutputParser()
questions = [
    {"Question0":
     """Situações envolvendo "DRONES" tornaram-se comuns, geralmente com intuito de facilitar a entrada de 
     ilícitos nos estabelecimentos penais. O Departamento de Controle do Espaço Aéreo - DECEA, 
     regulamentou o uso desses equipamentos. De acordo com o conteúdo apresentado, é permitida a circulação 
     desses veículos nas áreas de segurança?
    Escolha uma opção:
    a. Poderá ser utilizado em áreas de segurança desde que seja autorizado pelo detentor da instalação.
    b. Não é permitida e pode gerar penalidades severas.
    c. É permitida desde que seja feito por veículos de comunicação.
    d. É permitida desde que não cause prejuízo a unidade prisional.
     """
    },
    {"Question1":
     """Em caso de sobrevoo de aeronave, devemos ter atenção redobrada, haja vista, poder acontecer situações de 
        tentativa de arrebatamento por via aérea. Qual a altitude mínima de sobrevoo estabelecida pelos órgãos de 
        controle?
        Escolha uma opção:
        a. A aeronave deverá respeitar uma altitude mínima de 1.000 pés que equivale a cerca de 300 metros.
        b. A aeronave deverá respeitar uma altitude mínima de 2.000 metros.
        c. A aeronave poderá sobrevoar a unidade sem respeitar altitude mínima desde que não fique no mesmo local.
        d. A aeronave se tiver logotipo de identificação das policias, poderá sobrevoar a unidade sem a necessidade de aviso prévio."""
    },
    {"Question2":
     """Em situações de crise, existem duas hipóteses caracterizadas como excludentes de ilicitude, nas ações do 
        Policial Penal. Quais são elas?
        Escolha uma opção:
        a. Exercício regular de direito e legítima defesa.
        b. Legítima defesa e estrito cumprimento do dever legal.
        c. Estado de necessidade e exercício regular de direito.
        d. Legítima defesa e estado de necessidade."""
     }
]


response = chain.invoke(
    {
        "context": '\n'.join(doc.page_content for doc in documents),
        "question": questions[2],
    }
)

print(response)
