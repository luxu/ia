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
     """O serviço diário no Centro de Vigilância, exige do policial compromisso e dedicação na execução de suas 
        tarefas. Dentre elas, estão as que podem causar maior impacto negativo pela não obediência as 
        regras e normas vigentes. Dessa forma, quem é o responsável pela fiscalização dos policiais, 
        durante o turno de serviço?
        Escolha uma opção:
        a. Direto do Centro de Segurança e Disciplina.
        b. Diretor do Centro de Escolta e Vigilância Penitenciária.
        c. Diretor Técnico III.
        d. Diretor do Núcleo de Escolta e Vigilância Penitenciária.
     """
    },
    {"Question1":
     """Situações de anormalidade podem gerar inúmeros problemas à equipe de serviço. 
        Para que isso não ocorra, existe um plano a ser seguido de acordo com a Unidade Prisional à 
        qual o policial está lotado. Qual o nome desse plano?
        Escolha uma opção:
        a. Plano de Emergência.
        b. Plano de Contingência.
        c. Plano de Segurança
        d. Plano de Instrução."""
    },
    {"Question2":
     """Apesar de ser prática comum nas unidades prisionais, a utilização da mão de obra dos sentenciados 
        é prática inviável no Centro de Vigilância. Portanto, quem são os responsáveis pela higiene, conservação e 
        manutenção das instalações?
        Escolha uma opção:
        a. Os Policiais pertencentes aos Núcleos de Escolta e Vigilância.
        b. Qualquer integrante da Unidade Prisional.
        c. Presos que fazem parte do pavilhão de trabalho.
        d. Empresa de conservação e limpeza."""
     }
]


response = chain.invoke(
    {
        "context": '\n'.join(doc.page_content for doc in documents),
        "question": questions[2],
    }
)

print(response)
