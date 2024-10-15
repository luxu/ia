import os
from getpass import getpass

from decouple import config
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# pdf_path = "fit.pdf"
# pdf_path = "hr.pdf"
pdf_path = "mel.pdf"
# question = "Quanto foi o valor de desconto do Banco do Brasil? Toda resposta em português"
question = "Diabético pode consumir mel? Toda resposta em português"

os.environ["GOOGLE_API_KEY"] = config("API_GEMINI")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")


def load_pdf_in_chunks():
    # google_api_key = config("API_GEMINI")
    # genai.configure(
    #     api_key=google_api_key
    # )
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunks():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents=load_pdf_in_chunks())


def vector_store():
    # Load in vector
    persist_directory = "db"
    model_embedding = "models/text-embedding-004"
    # model_embedding = "models/embedding-001"
    embedding = GoogleGenerativeAIEmbeddings(model=model_embedding)
    chroma = Chroma.from_documents(
        documents=chunks(),
        embedding=embedding,
        persist_directory=persist_directory,  # Será persistido no disco
        collection_name="mel_info",
    )

    return chroma.as_retriever()


# model = "models/aqa"
model = "gemini-1.5-flash"
# model="gemini-1.5-pro"
prompt = hub.pull("rlm/rag-prompt")
retriever = vector_store()
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
# llm = ChatGoogleGenerativeAI(model=model)
# rag_chain = (
#     {
#         "context": retriever,
#         "question": RunnablePassthrough(),
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#
# response = rag_chain.invoke(question)
#
# print(response)
