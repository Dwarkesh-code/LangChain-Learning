from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
google = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
print(google.proflie)
embedding_model = OllamaEmbeddings(model="embeddinggemma:300m")
model = ChatGroq(model="llama-3.1-8b-instant")

print(model.profile)
print(embedding_model.profile)