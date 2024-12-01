import os
from getpass import getpass

from decouple import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

path_csv = "files/gasto_new.csv"

os.environ["GOOGLE_API_KEY"] = config("API_GEMINI")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")


def load_csv_in_chunks():
    loader = CSVLoader(path_csv)
    return loader.load()


def chunks():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents=load_csv_in_chunks())


def vector_store():
    # Load in vector
    persist_directory = "db"
    model_embedding = "models/text-embedding-004"
    # model_embedding = "models/embedding-001"
    # embedding = GoogleGenerativeAIEmbeddings(model=model_embedding)
    embedding = HuggingFaceEmbeddings()
    chroma = Chroma.from_documents(
        documents=chunks(),
        embedding=embedding,
        persist_directory=persist_directory,  # Será persistido no disco
        collection_name="gasto_csv",
    )

    return chroma.as_retriever()


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
