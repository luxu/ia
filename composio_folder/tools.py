# from langchain_core.prompts import ChatPromptTemplate
from decouple import config
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# list_model = [
#     'gemma-7b-it',
#     'llama3-groq-8b-8192-tool-use-preview',
#     'gemma2-9b-it',
#     'llama-3.1-70b-versatile',
#     'llama-3.2-90b-vision-preview',
# ]

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     api_key=config('API_GEMINI'),
# )
# model_embedding = "models/text-embedding-004"
# embedding = GoogleGenerativeAIEmbeddings(model=model_embedding)

class Tools:

    def __init__(self, tools=None, question=None, words_to_system=None):
        self.tools = tools
        self.words_to_system = words_to_system
        self.question = question

    # def get_llm(self):
    #     return ChatGroq(
    #         api_key=config('GROQ_API_KEY'),
    #         model=list_model[3]
    #     )

    def get_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=config('API_GEMINI'),
        )


    def get_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                ('system', self.words_to_system),
                ('human', '{input}'),
                ('placeholder', '{agent_scratchpad}'),
            ]
        )

    def get_agent(self):
        return create_tool_calling_agent(
            llm=self.get_llm(),
            tools=self.tools,
            prompt=self.get_prompt()
        )

    def get_agent_executor(self):
        return AgentExecutor(
            agent=self.get_agent(),
            tools=self.tools,
            verbose=True,
        )

    def get_result(self):
        return self.get_agent_executor().invoke({
            'input': self.question,
        })
