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
        Assinale a alternativa correta. De acordo com o conteúdo estudado, o alinhamento das viaturas 
        terá uma formação em:
        Escolha uma opção:
        a. Retaguarda-Retaguarda, exceto o Cerra-Fila que se manterá a frente do comboio e a esquerda.
        b. Todas alternativas estão corretas.
        c. Zigue-Zague, exceto o Cerra-Fila que se manterá no centro da via.
        d. Esquerda-Esquerda, exceto o Cerra-Fila que se manterá a direita.
         """
     },
    {"Pergunta1":
         """
        De acordo com o estudado neste bloco, assinale a alternativa incorreta:
        Escolha uma opção:
        a. A função do Cerra-Fila é realizar a segurança do comboio, desde o início até o término do comboio.
        b. A função dos balizadores é seguir as ordens do Cerra-Fila.
        c. O Carro-chefe e o Cerra-Fila utilizarão o dispositivo de iluminação intermitente diferente para se destacar 
        dos demais membros do comboio.
        d. A função do Carro-chefe é conduzir o comboio em segurança, passar as orientações por meio de comunicação 
        visual ou sonora quanto ao fechamento das vias ou mudança de faixa de rolamento desde o início do deslocamento 
        até o término.         """
     },
    {"Pergunta2":
         """
        Neste bloco de estudo sobre deslocamento em comboio, aprendemos sobre a composição mínima e máxima de um 
        comboio, e o que deve ser feito quando um comboio excede o número máximo de viaturas. Assinale a 
        alternativa correta:
        Escolha uma opção:
        a. Excedendo o limite máximo de dez viaturas um novo comboio deverá ser formado.
        b. Todas as alternativas estão corretas
        c. A composição máxima é dez viaturas.
        d. A composição mínima é três viaturas.
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
