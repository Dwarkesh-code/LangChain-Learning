from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
import subprocess 
import os 
import shutil
import atexit
from pathlib import Path


load_dotenv()

def remove_repo(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)



def extchecker(file_path):
    loader = None
    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pyc', '.ico']:
        return None
   
    if file_path.is_file() and ".git" not in file.parts:
        
        if file_path.suffix == '.pdf':
            print("py")
            loader = PyPDFLoader(file_path)

        elif file_path.suffix == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8')

        else:
            loader = TextLoader(file_path, encoding='utf-8')

    return loader

def get_splitter(ext, lang_map):
    lang= lang_map.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=800,
            chunk_overlap=150
        )
    
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

def get_trim(messages, model):
    trim_message = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",
    token_counter=model,
    start_on="human",
    include_system=True
    )
    return trim_message


def cleanup():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

def print_history(chat_history):
    print("\n\n",chat_history)


atexit.register(cleanup)
atexit.register(lambda: print_history(chat_history))
atexit.register(lambda: remove_repo(folder_name))

lang_map = {
    "cpp": Language.CPP,
    "go": Language.GO,
    "java": Language.JAVA,
    "kt": Language.KOTLIN,
    "js": Language.JS,
    "ts": Language.TS,
    "php": Language.PHP,
    "proto": Language.PROTO,
    "python": Language.PYTHON,
    "py": Language.PYTHON,
    "rst": Language.RST,
    "ruby": Language.RUBY,
    "rust": Language.RUST,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
    "markdown": Language.MARKDOWN,
    "md": Language.MARKDOWN,
    "latex": Language.LATEX,
    "html": Language.HTML,
    "sol": Language.SOL,
    "csharp": Language.CSHARP,
    "cobol": Language.COBOL
}

chat_history= [
    SystemMessage(content="You're an expert or helpful assistant that can explain every query base on context data and git hub repo data without halluciunate ")
]

file_chunks = []
folder_name = "temp_git_folder_repo"
repo_url = "https://github.com/Dwarkesh-code/LimitLens"
command=["git", "clone", repo_url, folder_name]


remove_repo(folder_name)


try:
    os.makedirs(folder_name, exist_ok=True)
    subprocess.run(command, check=True , capture_output=True ,text=True)
except subprocess.CalledProcessError as e :
    print("Error : ",e)
    

for file in Path(folder_name).rglob('*'):
    if file.is_file():
        loader = extchecker(Path(file))
        if loader is not None:
            ext = file.suffix.lstrip(".").lower()
            for doc in loader.lazy_load():
                splitter_fact= get_splitter(ext, lang_map)
                chunk = splitter_fact.split_documents([doc])
                file_chunks.extend(chunk) 

print("\nchunk file complete\n")


embedding_model = OllamaEmbeddings(model="embeddinggemma:300m")
model = ChatGroq(model="llama-3.1-8b-instant")
router_model = ChatGroq(model="llama-3.1-8b-instant")

vectorstore = Chroma.from_documents(
    documents=file_chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
print("\nVectorStore complete\n")


retriever = vectorstore.as_retriever(search_kwargs={"k":5})
print("\nRetriever complete\n")


router_prompt = PromptTemplate(
    template="""You're an expert that can summarize the query to specific key words,
    always give README.md in your answer,
    queries --- query answer. 
    explain this repo --- README.md or main file,
    explain this project --- project name,
    explain name project how it works what it do --- provide project name , and some more files ,
    explain how is this work --- extract exact work that user ask,
    You will answer only 'query answer' not make something else by Your self Try to not hallociunate and if You did't get valid answer give "README.md main file and files "

    Query or question : "{query}"
    """,
    input_variables=['query']
)

prompt = PromptTemplate(
    template="""You're the expert that can answer by the contest with exact query,
    chat history is a history that AI(You) and Human(user) perivious chat it will help You generate better response.But do not print the chat history okay.

    query: "{query}",

    context: \n"{context}",
    \n\n\n
    chat history : \n"{chat_history}" """,
    input_variables=[ 'query', 'context', 'chat_history']
)

parser = StrOutputParser()

while True:
   
    query = input("Your Query: ")

    if query == "exit":
        break

    router_query = (router_prompt| router_model | parser).invoke({"query":query})
    print("\nRouter Query done\n")

    context_docs = retriever.invoke(router_query)

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    chain = prompt | model | parser

    result = chain.invoke({"query": query, "context":context_text, "chat_history": chat_history})
    
    print(result)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result))
    chat_history = get_trim(chat_history, model)

    

print_history()

remove_repo(folder_name)
cleanup()