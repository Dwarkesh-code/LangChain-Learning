from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="gemma2:2b")

# prompt 1 -> topic
template_1  = PromptTemplate(
    template = "Write a complete research on this {topic}",
    input_variables=['topic']
) 

# prompt 2 -> summary of research topic
template_2 = PromptTemplate(
    template = "Write the 5 lines summary of this research \n {text}", 
    input_variables=['text'],
    validate_template=True
)


parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model | parser


result = chain.invoke({'topic':'Basic Of AI'})

print(result)
