from langchain_core.prompts import PromptTemplate


#template
template = PromptTemplate(
    template="""

Summarize the book titled "{book_input}" with the following specifictions:
Explanation style = "{style}"
Explanation Level = "{level}"

1. mathemetical answer: 
    try to add mathematical numbers examples like it's will improve this and it's can do this and You can earn this from this like example okay if examples are in this if not , not give example.

2. Don't try to summarise everything. Try to summarise key point what's the main point of book  

Explain it with setisfying requairments if Book not exist so directly say this book not exist. Do not hallocionate in any sitution and try to answer clear

""",
input_variables=['book_input', 'style', 'level'],
validate_template=True
)

template.save("template.json")