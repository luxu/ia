import os
import streamlit as st
import ollama

from decouple import config

from langchain import hub
# from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI

# os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')


def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()

    system_prompt = '''
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Responda em formato de markdown e com visualizações
    elaboradas e interativas.
    Contexto: {context}
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')


def use_streamlit():
    st.set_page_config(
        page_title='Estoque GPT',
        page_icon='📄',
    )
    st.header('Assistente Virtual')

    model_options = [
        'llama3.1:8b',
        'llava',
    ]

    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM',
        options=model_options,
    )

    st.sidebar.markdown('### Sobre')
    st.sidebar.markdown('Este agente pesquisa sobre diversos assuntos utilizando um modelo GPT.')

    st.write('Faça perguntas sobre qualquer assunto ligado a Tecnologia.')
    user_question = st.text_input('O que deseja saber?')

    question = st.chat_input('Como posso ajudar?')

    st.chat_message('user').write(question)

    prompt = '''
    Exercitando os conceitos de PLN utilizando NLTK, Spacy e Transformers
    Considerando as abordagens e exemplificações demonstradas no encontro de 05/10/2024:
    Segmentação
    Tokenização
    Tagger
    Stemmer
    Corpus
    Lematização
    Similaridade
    Desenvolva uma aplicação útil utilizado esses recursosA resposta final deve ter uma formatação amigável de visualização para o usuário.
    Sempre responda em português brasileiro.
    Pergunta: {q}
    '''
    prompt_template = PromptTemplate.from_template(prompt)
    if st.button('Consultar'):
        if user_question:
            with st.spinner('Buscando resposta...'):
                response = ask_question(
                    model=selected_model,
                    query=question,
                    # vector_store=vector_store,
                )
        else:
            st.warning('Por favor, insira uma pergunta.')


# model = ChatOpenAI(
#     model=selected_model,
# )
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# db = SQLDatabase.from_uri('sqlite:///db/chroma.sqlite3')
# toolkit = SQLDatabaseToolkit(
#     db=db,
#     llm=model,
# )
# system_message = hub.pull('hwchase17/react')

# agent = create_react_agent(
#     llm=model,
#     tools=toolkit.get_tools(),
#     prompt=system_message,
# )
#
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=toolkit.get_tools(),
#     verbose=True,
# )



def ollama_text():
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                'role': "user",
                'content': 'Qual a ling prog mais usada no mundo? Dê um exemplo médio',
            }
        ],
        stream=True
    )

    for chunk in response:
        print(chunk['message']['content'])

def ollama_image(filename):
    with open(filename, 'rb') as file:
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    'role': "user",
                    'content': 'O que tem na imagem?',
                    'images': [file.read()]
                },
            ],
            stream=True
        )
        for chunk in response:
            print(chunk['message']['content'], end='')


if '__main__' == __name__:
    use_streamlit()
    # ollama_text()
    filename = 'files/tatuagem.jpg'
    # ollama_image(filename)
