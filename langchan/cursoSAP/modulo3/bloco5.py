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
        Assinale a alternativa correta. De acordo com as normas de circulação e conduta, o artigo 61 no § 1º 
        diz que onde não existir sinalização regulamentadora, a velocidade máxima nas vias urbanas de trânsito 
        rápido será de:        
        Escolha uma opção:
        a. 80 km/h.
        b. 110 km/h.
        c. 100 km/h.
        d. 120 km/h.
         """
     },
    {"Pergunta1":
         """
        Neste bloco sobre Legislação de Trânsito, de acordo com o Código de Trânsito Brasileiro – C.T.B, no 
        artigo 1º e § 1º considera trânsito:
        Escolha uma opção:
        a. Nenhuma das alternativas estão corretas.
        b. Considera trânsito a utilização das vias por pessoas, veículos e animais, isolados ou em grupos, conduzidos 
        ou não, para fins de circulação, parada, estacionamento e operação de carga e descarga.
        c. Considera trânsito o bloqueio das vias para impedir a circulação de pessoas, veículos e animais, isolados 
        ou em grupos, conduzidos ou não, para fins de circulação, parada, estacionamento e operação de carga e descarga.
        d. Considera trânsito o fechamento total das vias.
         """
     },
    {"Pergunta2":
         """
        O artigo 167 do C.T.B trata sobre a obrigatoriedade do uso do cinto de segurança. Assinale a alternativa 
        correta:
        Escolha uma opção:
        a. Somente o condutor.
        b. É obrigatório o uso do cinto de segurança, tanto para o condutor como para os demais passageiros
        c. O uso é opcional.
        d. O uso é obrigatório somente em vias de trânsito rápido.
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
