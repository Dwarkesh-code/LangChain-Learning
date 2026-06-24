
import streamlit as st
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
from pathlib import Path
 
load_dotenv()
 
 
def remove_repo(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
 
 
def extchecker(file_path):
    loader = None
    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pyc', '.ico']:
        return None
 
    if file_path.is_file() and ".git" not in file_path.parts:
 
        if file_path.suffix == '.pdf':
            print("py")
            loader = PyPDFLoader(file_path)
 
        elif file_path.suffix == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8')
 
        else:
            loader = TextLoader(file_path, encoding='utf-8')
 
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
 
    \n\nAlso use meta data for best answer to retriever Meta data will  help You to answer the user query like how many files and which files are important and ignore the github or some waste file that not from repo.
 
    here is meta data : \n"{metadata}"
 
    """,
    input_variables=['query', 'metadata']
)
 
prompt = PromptTemplate(
    template="""You're the github repo expert that can answer by the contest with exact query,
    chat history is a history that AI(You) and Human(user) perivious chat it will help You generate better response.But do not print the chat history okay.
    Meta data will be also help You to answer the user query like how many files and which files are important and ignore the github or some waste file that not from repo.
 
    query: "{query}",
 
    context: \n"{context}",
    \n\n\n
    chat history : \n"{chat_history}" 
    
    \n\nMetaData: "{metadata}"
 
    """,
    input_variables=['query', 'context', 'chat_history', 'metadata']
)
 
parser = StrOutputParser()
 
 
# ---------- Streamlit cached resources (heavy objects, built once) ----------
 
@st.cache_resource
def get_models():
    embedding_model = OllamaEmbeddings(model="embeddinggemma:300m")
    model = ChatGroq(model="llama-3.1-8b-instant")
    router_model = ChatGroq(model="llama-3.1-8b-instant")
    return embedding_model, model, router_model
 
 
def build_vectorstore(repo_url, embedding_model):
    remove_repo(folder_name)
    command = ["git", "clone", repo_url, folder_name]
    os.makedirs(folder_name, exist_ok=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
 
    file_chunks = []
    for file in Path(folder_name).rglob('*'):
        if file.is_file():
            loader = extchecker(Path(file))
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
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20})
 
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
 
embedding_model, model, router_model = get_models()
 
with st.sidebar:
    st.header("Repo Setup")
    repo_url = st.text_input("GitHub repo URL")
    if st.button("Clone & Index Repo"):
        with st.spinner("Cloning repo and building vectorstore..."):
            try:
                retriever, metadata_set = build_vectorstore(repo_url, embedding_model)
                st.session_state.retriever = retriever
                st.session_state.metadata_set = metadata_set
                st.session_state.repo_loaded = True
                st.success("Repo indexed. Ask your questions below.")
            except subprocess.CalledProcessError:
                st.error("Enter valid Github Url")
 
    if st.button("Reset Conversation"):
        st.session_state.chat_history = [
            SystemMessage(content="You're an expert or helpful assistant that can explain every query base on context data and git hub repo data without halluciunate ")
        ]
        st.rerun()
 
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
 
    router_query = (router_prompt | router_model | parser).invoke(
        {"query": query, "metadata": st.session_state.metadata_set}
    )
 
    context_docs = st.session_state.retriever.invoke(router_query)
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
    st.info("Sidebar me GitHub repo URL daal kar 'Clone & Index Repo' click karo.")
