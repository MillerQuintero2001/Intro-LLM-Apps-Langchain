from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Create a document loader for fifa_countries_audience.csv
csv_path = script_dir / "fifa_countries_audience.csv"
loader = CSVLoader(str(csv_path))

# Load the document
data = loader.load()
print(data[0])

print(f"\n\nMetadata: {data[0].metadata}")