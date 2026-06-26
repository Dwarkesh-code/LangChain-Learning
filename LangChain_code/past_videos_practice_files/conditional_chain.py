from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

class Feddback(BaseModel):
    sentiment: Literal['Positive', 'Negative']

parser = PydanticOutputParser(pydantic_object=Feddback)

sentiment_prompt = PromptTemplate(
    template="Generate sentiment either postive or negative base on following feedback \n{feedback}\n{format_instruction}\nOutput must be in normal format like sentiment:Negative or Positive",
    input_variables=["feddback"],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

parser1 = StrOutputParser()

pos_prompt = PromptTemplate(
    template='Write down an appropriate response to this positive feedback \n{feedback}',
    input_variables=["feedback"]
)

neg_prompt = PromptTemplate(
    template='Write down an appropriate response to this negative feedback \n{feedback}',
    input_variables=["feedback"]
)


set_chain = sentiment_prompt | model | parser1

print(set_chain.invoke({"feedback":"This smartphone is terrible"}))

optional_chain = RunnableBranch(
    (lambda x:x == 'sentiment: Negative', neg_prompt | model | parser1),
    (lambda x:x == 'sentiment: Positive', pos_prompt | model | parser1),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = set_chain | optional_chain

result = chain.invoke({"feedback":"This smartphone is terrible"})

print(result)