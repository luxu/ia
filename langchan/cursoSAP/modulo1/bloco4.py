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
     """Em relação às transgressões disciplinares, o policial no posto de serviço deve ter sua atenção maximizada, 
     evitando qualquer tipo de distração. Desse modo, quais são os meios de comunicação, proibidos,
      nos postos de vigilância.
        Escolha uma opção:
        a. Jornais e revistas devem ser utilizados para reduzir a monotonia.
        b. O celular hoje é item essencial para desenvolver um bom trabalho.
        c. Aparelhos sonoros, telefones celulares, revistas, jornais, livros e similares que possam distrair atenção, 
        quando em exercício no posto de trabalho, exceto o rádio transceptor.
        d. Aparelhos sonoros desde que não atrapalhem o serviço.
     """
    },
    {"Question1":
     """Em relação aos deveres e obrigações policiais, devemos nos apresentar para o serviço de 
     acordo com a legislação vigente. 
        Escolha uma opção:
        a. Assumir o serviço com roupa paisana.
        b. Utilizar os equipamentos que adquiri em curso terceirizado.
        c. Assumir o posto de serviço sem me preocupar com horário.
        d. Apresentar-se em serviço, devidamente, uniformizado, asseado, barbeado e com postura adequada."""
    },
    {"Question2":
     """Em caso de perda ou extravio de material bélico, pertencente ao Centro de Vigilância, 
     qual a sequência de ações referentes à escrituração devem ser seguidas?
        Escolha uma opção:
        a. Relato no Livro de Ocorrências, Confecção de Boletim de Ocorrência e Comunicado de Evento
        b. Somente Boletim de Ocorrência.
        c. Somente Comunicado de Evento.
        d. Somente relato no Livro de Ocorrências."""
     }
]


response = chain.invoke(
    {
        "context": '\n'.join(doc.page_content for doc in documents),
        "question": questions[0],
    }
)

print(response)
