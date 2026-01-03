from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Create a document loader for fifa_countries_audience.csv
csv_path = script_dir / "fifa_countries_audience.csv"
loader = CSVLoader(str(csv_path))

# Load the document
data = loader.load()

print(f"First 3 rows of the CSV, content and metadata:\n")
for i in range(3):
    print(f"Content:{data[i].page_content}")
    print(f"\nMetadata: {data[i].metadata}")
    print("\n" + "-"*50 + "\n")