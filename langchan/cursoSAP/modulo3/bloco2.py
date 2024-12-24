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
        Assinale a alternativa incorreta quanto ao câmbio automático. sendo a viatura a SW4 Hilux.
        Escolha uma opção:
        a. “P” PARKING – é o ponto que você deixa o carro quando vai pará-lo.
        b. “D” DRIVE – é a posição utilizada para o veículo se movimentar
        c. “N” NEUTRAL – é a posição usada para subidas/descidas inclinadas.
        d. “R” REVERSE – é usada para sair em Ré.
         """
     },
    {"Pergunta1":
         """
        De acordo com o vídeo de dicas sobre o Câmbio Automático, quais ações que devem ser realizadas pelo condutor:
        Escolha uma opção:
        a. Pare o carro, após pise no pedal do freio pra trocar do drive “D” e colocar na marcha Ré “R”.
        b. Nunca engate o “D” ou “R” com o carro em movimento.
        c. Todas as alternativas estão corretas.
        d. Nunca coloque o carro em Parking “P” antes que ele pare totalmente.
         """
     },
    {"Pergunta2":
         """
        Quanto ao PADDLE SHIFT, assinale a alternativa correta.
        Escolha uma opção:
        a. Todas as alternativas estão corretas.
        b. Para acionar essa função, basta puxar qualquer um dos PADDLES em sua direção.
        c. Para aumentar basta acionar o PADDLE SHIFT UP (+) lado direito, e para diminuir basta acionar o PADDLE SHIFT DOWN (-) lado esquerdo.
        d. Para retornar a posição “D” basta acionar o PADDLE SHIFT UP (+) em sua direção e segurar por 2 segundos.
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
