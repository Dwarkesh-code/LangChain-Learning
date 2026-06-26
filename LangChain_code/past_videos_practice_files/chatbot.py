from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

chat_history = [
    SystemMessage(content="You are the best reasoning model in the world")
]

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)