import os

from decouple import config

from langchain_groq import ChatGroq

from langchain.chains import create_extraction_chain
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

groq_api_key = config('GROQ_API_KEY')


llm = ChatGroq(
    model='llama-3.1-70b-versatile',
    api_key=groq_api_key,
)

def extract(content, schema):
    return create_extraction_chain(
        llm=llm,
        schema=schema
    ).invoke(content).get('text')


def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformed = BeautifulSoupTransformer()
    docs_transformed = bs_transformed.transform_documents(
        documents=docs,
        tags_to_extract=["table"],
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=0,
    )
    splits = splitter.split_documents(
        documents=docs_transformed
    )
    extracted_content = []

    for split in splits:
        extracted_content.extend(
            extract(
                content=split.page_content,
                schema=schema,
            )
        )
    return extracted_content

if __name__ == "__main__":
    schema = {
        "properties": {
            "posicao": {"type": "integer"},
            "time": {"type": "string"},
            "jogos": {"type": "integer"},
            "vitorias": {"type": "integer"},
            "empates": {"type": "integer"},
            "derrotas": {"type": "integer"},
            "gols_pro": {"type": "integer"},
            "gols_contra": {"type": "integer"},
            "saldo_gols": {"type": "integer"},
            "pontos": {"type": "integer"},
        },
    }
    urls = [
        'https://ge.globo.com/futebol/brasileirao-serie-a/',
    ]
    extracted_content = scrape_with_playwright(urls, schema)
    print(extracted_content)
