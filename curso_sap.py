import streamlit as st
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
from groq import Groq

groq_api_key = config('GROQ_API_KEY')


def ask_question(query):
    # llm = ChatGroq(
    #     temperature=0.4,
    #     model_name=model,
    #     api_key=groq_api_key
    # )
    client = Groq(
        api_key=groq_api_key
    )
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": "Obrigado por perguntar!\n\nÉ importante notar que a letalidade de uma arma depende de várias variáveis, como sua utilização, a habilidade do usuário e as circunstâncias em que é usada. Além disso, não há uma resposta única para essa pergunta, pois diferentes armas podem ser extremamente perigosas em diferentes contextos.\n\nQue são algumas das armas mais letais do mundo:\n\n1. **Sniper rifle:** Essas armas de fogo longo são projetadas para disparar munição de precisão a grandes distâncias. Elas podem ser usadas para atacar alvos movediços e estáticos a partir de distâncias significantes.\n2. **Machine gun:** Armas de fogo rápido podem causar danos massivos em um curto período de tempo. Eles são frequentemente usados em conflitos em larga escala.\n3. **Nuclear weapons:** As armas nucleares são consideradas as mais letais do mundo, pois podem causar danos imensos e perdas humanas em grandes áreas.\n4. **Hand grenades:** Essas armas podem causar danos letais em áreas fechadas e podem ser usadas em combinação com outras armas.\n5. **Guided missiles:** Foguetes guias são projecteis direcionados que podem ser usados para atacar alvos movimentados a partir de distâncias significativas.\n\nÉ importante lembrar que a letalidade de uma arma não é necessariamente igual à sua eficácia em batalha. Além disso, a possibilidade de uso indevido de armas pode causar danos significativos a pessoas inocentes e ao meio ambiente.\n\nÉ fundamental ressaltar que a segurança é o primeiro passo para a prevenção de perdas humanas e danos ao meio ambiente."
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    breakpoint()
    return completion


    # system_prompt = '''
    # Use o contexto para responder as perguntas.
    # Se não encontrar uma resposta no contexto,
    # explique que não há informações disponíveis.
    # Responda em formato de markdown e com visualizações
    # elaboradas e interativas.
    # Contexto: {context}
    # '''
    # messages = [('system', system_prompt)]
    # for message in st.session_state.messages:
    #     messages.append((message.get('role'), message.get('content')))
    # messages.append(('human', '{input}'))
    #
    # prompt = ChatPromptTemplate.from_messages(messages)

    return llm.invoke({'input': query})

model='llama3-70b-8192'

st.set_page_config(
    page_title='SAP GPT',
    page_icon='📄',
)
st.header(
    'Assistente Virtual das questões do curso SAP'
)
st.sidebar.markdown('### Sobre')
st.sidebar.markdown(
    'Qual a maior linguagem do mundo?'
)
question = st.chat_input('Como posso ajudar?')
user_question = st.text_input('O que deseja saber?')
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
st.chat_message('user').write(question)
# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")

if st.button('Consultar'):
    if user_question:
        with st.spinner('Buscando resposta...'):
            response = ask_question(query=question)
    else:
        st.warning('Por favor, insira uma pergunta.')
