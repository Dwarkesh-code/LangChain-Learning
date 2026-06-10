from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser 
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field 

load_dotenv()

model = ChatOllama(model="gemma2:2b", format="json")


class Person(BaseModel):

    name : str = Field(description="Name of the person")
    age : int = Field(gt=18, description="Age of the person")
    city : str = Field(description="City of the person")


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="Genrate a fiction person from {place} \n {format_instruction} \n Give JSON format that have name, age and city |  don't give nothing else from json    ",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

#user = input("Enter country name : ")

prompt = template.invoke({'place':"Indian"})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)