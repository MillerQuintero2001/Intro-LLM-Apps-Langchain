from langchain_community.document_loaders import UnstructuredHTMLLoader
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Create a document loader for rag_vs_fine_tuning.pdf
html_path = script_dir / "white_house_executive_order_nov_2023.html"
loader = UnstructuredHTMLLoader(str(html_path))

# Load the document
data = loader.load()
print(f"Content: '{data[0].page_content}'")
print(f"\n\nMetadata: {data[0].metadata}")