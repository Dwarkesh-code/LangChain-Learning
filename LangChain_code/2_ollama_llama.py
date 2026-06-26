from langchain_ollama import ChatOllama 

llm = ChatOllama(model="qwen2.5:3b") 

result = llm.invoke("Who is the worlds best CLI model for free")

print(result.content)