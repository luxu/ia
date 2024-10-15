import os
from getpass import getpass
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = config("API_GEMINI")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")

persist_directory = "db"
query = "Diabético pode consumir mel? Toda resposta em português"
model_embedding = "models/text-embedding-004"

embedding = GoogleGenerativeAIEmbeddings(model=model_embedding)

vector_store = Chroma(
    persist_directory=persist_directory,  # Será persistido no disco
    embedding_function=embedding,
    collection_name="mel_info",
)

retriever = vector_store.as_retriever()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
system_prompt = """
Use o contexto para responder as perguntas.
Contexto: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

response = chain.invoke({"input": query})

print(response)
