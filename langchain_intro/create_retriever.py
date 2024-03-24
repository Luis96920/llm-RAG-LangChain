# Imports
import dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings

# Constants. These are the paths to the data files.
REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

# Load the environment variables
dotenv.load_dotenv()

# Load the reviews from the CSV file using the CSVLoader
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

# Create a ChromaDB instance. This will store the embeddings of the reviews.
reviews_vector_db = Chroma.from_documents(
    reviews, OpenAIEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH
)