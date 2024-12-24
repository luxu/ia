import streamlit as st
from decouple import config
from langchain_community.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

loader = CSVLoader(
    file_path="../files/gasto_new.csv",
)
documents = loader.load()
# embeddings = OpenAIEmbeddings()
# model_embedding = "models/text-embedding-004"
# embeddings = GoogleGenerativeAIEmbeddings(model=model_embedding)
embeddings = HuggingFaceEmbeddings()  # embedding FREE
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]


question = "Quais comércios mais aparecem na planilha... Toda resposta em português"
response = retrieve_info(question)
print(response)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=config('API_GEMINI'),
)
template=f"""Você é um assistente responsável por fazer a gestão do meus gastos mensais e deve olhar para a minha 
planilha e responder às perguntas e dar dicas. {question}"""
prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

print(generate_response("Maiores gastos e data deles... Toda resposta em português"))