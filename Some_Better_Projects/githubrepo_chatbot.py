import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from typing import Literal, Optional
from pydantic import BaseModel, Field
import subprocess
import os
import shutil
import atexit
from pathlib import Path
 
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
 
def remove_repo(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
 
 
def extchecker(file_path, lang_map):
    loader = None
    if file_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.pyc', '.ico']:
    
        if file_path.is_file() and ".git" not in file_path.parts:
    
            if file_path.suffix[1:] in list(lang_map.keys()) or file_path.suffix in ['.md','.txt']:
                loader = TextLoader(file_path)
    
    return loader
 
 
def get_splitter(ext, lang_map):
    lang = lang_map.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=1500,
            chunk_overlap=300
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
 
atexit.register(cleanup)
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
 
folder_name = "temp_git_folder_repo"


class RouterStructure(BaseModel):
    retriever_query : list[str] = Field(description="Generate 3 queries base on explanation of query. When user say explain(summarize) repo/project So provide specific words like readme, project name, topic name from readme etc.")
    retriever_keywords : list[str] = Field(description="Provide keywords from queries and always provide atleast 1 keyword for readme base on query ")
    k : Optional[int] = Field(default=10, ge=8, le=50, description="Provide a integer chunk value (3-15) base on user query difficulty. 1300 to 1500 charactors are in a chunk. Based on the number of chunks required to resolve the query, provide an integer value between 8 and 50.")
    fetch_k : Optional[int] = Field(default=18, ge=31, le=100, description="The size of the initial pool of documents for MMR search in retriever. CRITICAL RULE: This value must ALWAYS be greater than or equal to 'k'. Recommended value: k + 10 (or at least 15). Provide an integer between 31 and 100.")


router_prompt = PromptTemplate(
    template="""
    query : {query},\n

    some examples 
    > explain/summarize repo/project and rate this repo --> provide readme data and some imp projects data and provide k(15-29) and fetch_k(40 - 60).
    > explain this function and bug --> provide specific data of function and  base on specific data provide k and fetch_k
    > if in query user say use your max pontential or use simmilar keywords like read every file, use max, deep and some more --> so give k(50)  fetch_k(100)

    for projects and file name are in meta data
    \nMetaData : {metadata}
""",
    input_variables=['query', 'metadata']
)

 
prompt = PromptTemplate(
    template="""You're the github repo expert that can answer user query with context data and meta data,
    
    Chat history alternates between user turns and your previous responses.
   
    MetaData provides You projects and files name.

    


    Some examples: 
    1. summarize/explain this repo --- readme.md and some main files.
    2. Summarize/explain this project --- Generate response from context data in context data you will get project detail if there none detail in conext data and metadata about project directly say there is no project of this name.
    3. Rate this repo --- First read some files readme and context data then Rate the repo between 1 ⭐ and 5 ⭐ based on code quality, structure and documation structure with honest. Don't change Your rating in any situation. Also Try to give madels for repo (🎖️,🏅,🥇,🥈,🥉) 

    Strict Rules:-
    1. Don't use prompt language for generating response
    2. Try to Not use words like HumanMessages, AiMessages, Context Data, Meta data and some more from prompt.
    3. Don't hallucinate in any situation.
    4. If You don't know say "I don't know", "I can't able to understand Your query", "I don't have enough data from repo" ETC.
    5. Don't provide rating or madels until in query user wants the rating of repo.
    6. Don't generate response for extra things from Your own thinking. only generate response that user want.
    7. rule for Developer information :-
        1. Don't Use Developer information until user ask You about developer
 
    query: "{query}",
 
    context: \n"{context}",
    \n\n\n
    chat history : \n"{chat_history}" 
    
    \n\nMetaData: "{metadata}"

    \nSome Informations about You and Your developer :- 
    1. You're GitHub Repo ChatBot
    2. Your main LLM is 'llama-3.1-8b-instant' and Your api provider groq
    3. Your developer is Dwarkesh code a 16-17 year old teenager.
    4. Dwarkesh code's Github --> https://github.com/Dwarkesh-code
    5. Dwarkesh code's Linkedin --> www.linkedin.com/in/dwarkesh-code
    6. He (Dwarkesh code) built Limit Lens(claude usage tracer) a chrome browser extension that can trace claude's usages and he is learning langchain framework.
    
    
    
    """,
    input_variables=['query', 'context', 'chat_history', 'metadata']
)
 
parser = StrOutputParser()
 
 
# ---------- Streamlit cached resources (heavy objects, built once) ----------
 
@st.cache_resource
def get_models():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    router_model = ChatGroq(model="qwen/qwen3-32b")
    return embedding_model, model, router_model
 
 
def build_vectorstore(repo_url, embedding_model, lang_map):
    remove_repo(folder_name)
    command = ["git", "clone", repo_url, folder_name]
    os.makedirs(folder_name, exist_ok=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
 
    file_chunks = []
    for file in Path(folder_name).rglob('*'):
        if file.is_file():
            loader = extchecker(Path(file), lang_map)
            if loader is not None:
                ext = file.suffix.lstrip(".").lower()
                for doc in loader.lazy_load():
                    splitter_fact = get_splitter(ext, lang_map)
                    chunk = splitter_fact.split_documents([doc])
                    file_chunks.extend(chunk)
 
    vectorstore = Chroma.from_documents(
        documents=file_chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
 
    clean_metadata = []
    for doc in vectorstore.get(include=['metadatas']).get('metadatas', []):
        source = doc.get('source', '').replace('temp_git_folder_repo/', '')
        clean_metadata.append(source)
 
    metadata_set = list(set(clean_metadata))
 
    retriever = vectorstore.as_retriever(
        search_type="mmr")
 
    return retriever, metadata_set
 
 
# ---------- Streamlit session state ----------
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You're an expert or helpful assistant that can explain every query base on context data and git hub repo data without halluciunate ")
    ]
 
if "retriever" not in st.session_state:
    st.session_state.retriever = None
 
if "metadata_set" not in st.session_state:
    st.session_state.metadata_set = None
 
if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False
 
 
# ---------- UI ----------
 
st.title("GitHub Repo Chatbot")
st.write("By Dwarkesh Code")
 
embedding_model, model, router_model = get_models()

structure_router_model = router_model.with_structured_output(RouterStructure)


with st.sidebar:
    st.header("Repo Setup")
    repo_url = st.text_input("GitHub repo URL")
    if st.button("Clone & Index Repo"):
        with st.spinner("Cloning repo and building vectorstore..."):
            try:
                cleanup()
                remove_repo(folder_name)
                retriever, metadata_set = build_vectorstore(repo_url, embedding_model, lang_map)
                st.session_state.retriever = retriever
                st.session_state.metadata_set = metadata_set
                st.session_state.repo_loaded = True
                st.success("Repo indexed. Ask your questions below.")
            except subprocess.CalledProcessError:
                st.error("Enter valid Github Url")
 
 
# Display existing chat history (skip the SystemMessage)
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
 
query = st.chat_input("Your Query:", disabled=not st.session_state.repo_loaded)
 
if query:
    with st.chat_message("user"):
        st.write(query)
 
    router_query = (router_prompt | structure_router_model).invoke(
        {"query": query, "metadata": st.session_state.metadata_set}
    )
    
    retriever_prompt = f"{router_query.retriever_query} \n{router_query.retriever_keywords}"

    if router_query.fetch_k <router_query.k:
        router_query.fetch_k = router_query.k + 10

    context_docs = st.session_state.retriever.invoke(
        retriever_prompt,
        search_kwargs={"k":router_query.k, "fetch_k": router_query.fetch_k} )
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
 
    chain = prompt | model | parser
 
    result = chain.invoke({
        "query": query,
        "context": context_text,
        "chat_history": st.session_state.chat_history,
        "metadata": st.session_state.metadata_set
    })
 
    with st.chat_message("assistant"):
        st.write(result)
        with st.expander("Router query"):
            st.write(router_query)
 
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=result))
    st.session_state.chat_history = get_trim(st.session_state.chat_history, model)
 
if not st.session_state.repo_loaded:
    st.info("Enter GitHub url in the sidebaar and click 'clone & index repo' ")
