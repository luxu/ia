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

loader = PyPDFLoader("../../../files/modulo2/material.pdf")
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
questions = [
    {"Pergunta0":
     """Qual é o traje adequado para a Equipe AEVP designada a realizar a Escolta Interestadual Aérea de Presos?
        Escolha uma opção:
        a. Trajes de gala.
        b. Uniforme completo AEVP.
        c. Trajes civis.
        d. Uniforme incompleto AEVP."""
    },
    {"Pergunta1":
     """Dado o risco elevado envolvido na movimentação externa de presos que necessitam de tratamento de hemodiálise, 
        o que é essencial fazer antes que o preso embarque novamente no veículo de transporte para retornar 
        à unidade prisional?
        Escolha uma opção:
        a. Fazer perguntas ao preso para determinar se ele tem intenções maliciosas.
        b. Realizar uma busca visual.
        c. Realizar uma busca pessoal minuciosa.
        d. Vistoriar os pertences do preso"""
    },
    {"Pergunta2":
     """Durante o velório, o preso deverá permanecer isolado das demais pessoas, e o tempo que ele permanecerá 
        isolado no local será previamente definido pelo responsável pela escolta. De acordo com o Procedimento 
        Operacional Padrão (POP), qual é o tempo permitido para essa permanência?
        Escolha uma opção:
        a. Mínimo de 5 minutos.
        b. Máximo de 15 minutos e mínimo de 5 minutos.
        c. Máximo de 15 minutos.
        d. Máximo de 5 minutos."""
     }
]


response = chain.invoke(
    {
        "context": '\n'.join(doc.page_content for doc in documents),
        "question": questions[2],
    }
)

print(response)
