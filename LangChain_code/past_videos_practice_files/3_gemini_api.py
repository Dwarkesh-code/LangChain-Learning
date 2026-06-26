from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=10000
    )

result = llm.invoke("Who is the Narendar Modi")

print(result.content)
print("---")
print(result.usage_metadata)