from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(
    template="Generate summary base on Document : \n{docs} \n If Document is Nothing return 'There is No Document'.\nIn documents may be You get csv type or txt type file, generate Your answer base on file type  ",
    input_variables=['docs']
)



def extchecker(file_path):
    loader = None
    if file_path in ['.pdf', '.csv', '.txt']:
        
        if file_path.is_file():
            
            if file_path == '.pdf':
                loader = PyPDFLoader(file_path, encoding='utf-8-sig')
            
            elif file_path == '.txt':
                loader = TextLoader(file_path, encoding='utf-8-sig')

            elif file_path == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8-sig')
            
    return loader


file_path = Path(input("Enter Your File Path : ").strip())

if file_path.is_dir():
    for file in file_path.iterdir():
        loader = extchecker(file)

else :
    loader = extchecker(file_path)


if loader:
    docs = loader.load()
else : 
    docs = ['Nothing']

parser = StrOutputParser() 

chain = prompt | model | parser

result = chain.invoke({'docs':docs})

print(result)