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
            "level": {"type": "string"},
            "dps": {"type": "string"},
            "dpa": {"type": "string"},
            "hitpoints": {"type": "string"},
            "ability level": {"type": "string"},
            "regeneration time": {"type": "string"},
            "training cost": {"type": "string"},
            "training time": {"type": "string"},
            "required town hall": {"type": "string"},
        },
    }
    urls = [
        'https://clashofclans.fandom.com/wiki/Minion_Prince',
    ]
    extracted_content = scrape_with_playwright(urls, schema)
    print(extracted_content)
