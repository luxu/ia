import pprint
from typing import Dict, Any

from decouple import config
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from pandas import read_csv

groq_api_key = config('GROQ_API_KEY')

# df_query = "Retrieve the average of the valor_parcela column from rows 1 to 3."
df_query = "Média da coluna valor_parcela das linhas 1 a 3."

def format_parser_output(parser_output: Dict[str, Any]) -> None:
    for key in parser_output.keys():
        parser_output[key] = parser_output[key].to_dict()
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)

llm = ChatGroq(
    model='llama-3.1-70b-versatile',
    api_key=groq_api_key,
)
df = read_csv("gastosluxu.csv")
parser = PandasDataFrameOutputParser(dataframe=df)

prompt = PromptTemplate(
    template="Responder à consulta do usuário.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | llm | parser
parser_output = chain.invoke({"query": df_query})

format_parser_output(parser_output)
