import os

from decouple import config
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_groq import ChatGroq

groq_api_key = config('GROQ_API_KEY')


if __name__ == '__main__':
    llm = ChatGroq(
        model='llama-3.1-70b-versatile',
        api_key=groq_api_key,
    )
    browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(
        sync_browser=browser,
    )
    tools = toolkit.get_tools()

    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    result = agent_chain.invoke(
        input='qual time está na primeira posição do brasileirão na tabela do site https://ge.globo.com/futebol/brasileirao-serie-a/? e o último colocado'
    )
    print(result.get('output'))
