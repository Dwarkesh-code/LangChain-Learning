from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate, load_prompt


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

st.header("Web Page AI Model")

#inputs 
book_input = st.selectbox("Select Book Name", ["The Psychology of Money by Morgan Housel", "Deep Work by Cal Newport", "Mindset: The New Psychology of Success by Carol S. Dweck", "Atomic Habits by James Clear", "Think and Grow Rich by Napoleon Hill"])

explanation_style = st.selectbox("Select Explanation Style", ["Fun Way", "Serious Way", "Motivate Way", "Love way", "Power Way"])

explanation_level = st.selectbox("Select Explanation level", ["low", "medium", "extreme"])

# load template
template = load_prompt("template.json")

# AI massage 
if st.button("Summarise"):
    chain = template | llm
    result = chain.invoke({
        'book_input':book_input,
        'style': explanation_style,
        'level': explanation_level
    })
    st.write(result.content) 
