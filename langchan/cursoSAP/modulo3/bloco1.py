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
        Assinale a alternativa correta. De acordo com a imagem de calibragem dos pneus, 
        é correto afirmar que a pressão alta dos pneus:
        Escolha uma opção:
        a. Aumenta o desgaste no meio dos pneus.
        b. Aumenta a sua vida útil.
        c. Facilita as manobras.
        d. Traz maior conforto aos ocupantes do veículo.
         """
     },
    {"Pergunta1":
         """
        De acordo com o conteúdo estudado neste bloco, o responsável pela manutenção de 1º escalão da viatura é o:
        Escolha uma opção:
        a. Condutor da viatura.
        b. De ninguém, pois não é necessário fazer a manutenção de 1º escalão.
        c. Encarregado e Sub portaria.
        d. Somente o Setor de frota.
         """
     },
    {"Pergunta2":
         """
        O que previne a manutenção de 1º escalão? Assinale a alternativa correta.
        Escolha uma opção:
        a. Nenhuma das alternativas.
        b. O funcionamento dos airbag.
        c. Previne o esterçamento correto do volante numa curva acentuada.
        d. Pane elétrica; Pane mecânica; Pane seca; e Superaquecimento do motor.
         """
     }
]
choice_questions = questions[2]


loader = PyPDFLoader(file)
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Seja um aluno da EAP, respondendo perguntas sobre o conteúdo ministrado em aula.
        Não use a internet para procurar e se baseie no material passado caso possível.
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
