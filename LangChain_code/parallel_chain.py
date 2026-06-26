from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
qwen = ChatGroq(model="qwen/qwen3-32b")


notes_prompt = PromptTemplate(
    template="Generate the deep notes with example on {topic}",
    input_variables=['topic']
)

quiz_prompt = PromptTemplate(
    template="Generate 5 good level quiz on {topic}",
    input_variables=['topic']
)

merge_prompt = PromptTemplate(
    template="Combine the following notes and quiz into a single structured study document. First present the notes, then the quiz below them. \n{notes}  and  \n{quiz}",
    input_variables=['notes', 'quiz']   
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': notes_prompt | gemini | parser ,
    'quiz': quiz_prompt | qwen | parser
})

merge_chain = merge_prompt | gemini | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"topic":"Explain Paralllel Chain topic in LangChain"})

print(result)

chain.get_graph().print_ascii()

