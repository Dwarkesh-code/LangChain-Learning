from past_videos_practice_files.docloader import extchecker
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader 
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path 


char_splitter = {
    'character':CharacterTextSplitter,
    'rescursive':RecursiveCharacterTextSplitter
}

file_path = Path(input('Your file path or text: ').strip())

loader = extchecker(file_path)
text = loader.load() if loader else str(file_path)

user = input(f"Enter Splitter {list(char_splitter.keys())} : ")
size = int(input("chunk size: "))
overlap = int(input("chunk  overlap: "))

before_result = char_splitter[user](chunk_size=size, chunk_overlap=overlap)

result = before_result.split_text(text)if isinstance(text, str) else  before_result.split_documents(text)

print("Length: ",len(result))
print("Result: ", result)
#print("AVG Chunk size: ",result.chunk)
