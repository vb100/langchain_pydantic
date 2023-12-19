import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
    )
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

api_key = os.environ['OPENAI_API_KEY']
model = ChatOpenAI(openai_api_key=api_key, temperature=0)

# ---- DEFINE OUTPUT DATA TYPES WITHIN THE CLASS AND INITILIAZE A PARSER ----
class Players(BaseModel):

    values: list = Field(description='Python list of dictionaries containing player name and nationality')
    city: str = Field(description='Give me the most popular country across the results')

parser = PydanticOutputParser(pydantic_object=Players)

# ---------------------------- SETUP THE REQUEST -----------------------------
human_prompt = HumanMessagePromptTemplate.from_template("{request}\n{format_instructions}")
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

request = chat_prompt.format_prompt(
    request='Give me facts about 100 NBA players around the world',
    format_instructions=parser.get_format_instructions()
).to_messages()

results = model(request, temperature=0)
results_values = parser.parse(results.content)  # Player class object


# ----------------------------- SHOW THE RESULTS -----------------------------
import pandas as pd

results_dataframe = pd.DataFrame.from_dict(results_values.values)

print(results_dataframe.head(10))
print(results_dataframe.shape)

print(f'The most popular city across the results is {results_values.city}')

# ------------------------------------------------------------------------------
# Learn data science from industry experts and prepare real-world projects!
# CHECK IT: https://turingcollege.org/DataScienceGarage
#
# Turing College 
# ------------------------------------------------------------------------------
