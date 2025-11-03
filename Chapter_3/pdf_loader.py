from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Create a document loader for rag_vs_fine_tuning.pdf
pdf_path = script_dir / "rag_vs_fine_tuning.pdf"
loader = PyPDFLoader(str(pdf_path))

# Load the document
data = loader.load()
print(data[0])

print(f"\n\nMetadata: {data[0].metadata}")