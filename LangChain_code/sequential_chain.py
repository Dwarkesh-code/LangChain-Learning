from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


prompt1 = PromptTemplate(
    template="Generate the research on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Create 5 line summary of this text \n{text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser

result = chain.invoke({'topic': "How AI is dangerous"})

print(result)

chain.get_graph().print_ascii()