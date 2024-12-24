from decouple import config

import pandas as pd
from pandasai import SmartDataframe

from langchain_groq.chat_models import ChatGroq


groq_api_key = config('GROQ_API_KEY')
llm = ChatGroq(temperature=0.4, model_name = 'llama3-70b-8192', api_key = groq_api_key)
data = pd.read_csv("gastosluxu.csv")
print(data.head())
df = SmartDataframe(data,config={'llm': llm})
# df.chat('scatter plot for the composition of copper and Lead')
# df.chat('List out the columns in the dataset as table format')
# df.chat('Find the average of the column named valor_parcela')
# result = df.chat('Group and add the valor_parcela taking the same id_expenses and showing the result in table format and return too column name')
result = df.chat('List columns: name, datagasto, valor_parcela in the dataset as table format')
print(result)
