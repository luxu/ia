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

loader = PyPDFLoader("../../files/bloco2.pdf")
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
     """De acordo com a Resolução SAP-89, o Núcleo de Escolta e vigilância tem atribuições bem definidas, 
        sendo duas delas as principais. Assinale a alternativa correta.
        Escolha uma opção:
        a. Guarda dos presos que trabalham na parte externa e o cumprimento de suas atividades.
        b. Manutenção da muralha e limpeza da subportaria.
        c. Conduzir o veículo de transporte de presos e conferir documentação relativa aos visitantes 
        da unidade prisional.
        d. Exercer a vigilância armada nas muralhas, alambrados e guaritas da unidade prisional e exercer a 
        escolta armada, vigilância e proteção dos presos, quando em trânsito e movimentação externa.
     """
    },
    {"Question1":
     """A hierarquia é conceito básico em todas as instituições, sejam elas públicas ou privadas. 
        Deste modo, podemos definir hierarquia como. Assinale a alternativa correta.
        Escolha uma opção:
        a. É a relação de subordinação existente entre os vários órgãos e agentes do executivo com a distribuição de 
        função e a gradação da autoridade de cada um.
        b. A hierarquia só traz prejuízos ao trabalho policial.
        c. A hierarquia deveria ser abolida das instituições públicas.
        d. A hierarquia tem definições apenas para instituições militares."""
    },
    {"Question2":
     """São fundamentais para a segurança, além da figura do Policial Penal, outros meios para auxiliá-lo. 
        Assinale a alternativa correta. NO alarme, que faz parte dessas ferramentas, 
        que é classificado em dois tipos: quais são eles?
        Escolha uma opção:
        a. Sistema Central e Sistema por Rádio Comunicador.
        b. Sistema de Disparo de Arma de Fogo e Alarme Local.
        c. Sistema Local e Sistema com Cães.
        d. Sistema de Alarme Local e Sistema de Alarme de Posto Central."""
     }
]


response = chain.invoke(
    {
        "context": '\n'.join(doc.page_content for doc in documents),
        "question": questions[2],
    }
)

print(response)
