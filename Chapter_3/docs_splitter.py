from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Load the HTML document into memory
html_path = script_dir / "white_house_executive_order_nov_2023.html"
loader = UnstructuredHTMLLoader(str(html_path))
data = loader.load()

# Define variables
chunk_size = 300
chunk_overlap = 100

# Split the HTML
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=['.'])

docs = splitter.split_documents(data)
print(docs[0])
print("\n\nLa longitud de cada chunk es:")
print([len(doc.page_content) for doc in docs])