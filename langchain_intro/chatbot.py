import dotenv # Import the dotenv package to load environment variables

from langchain_openai import ChatOpenAI # Import the ChatOpenAI class from langchain_openai package

dotenv.load_dotenv() # Load environment variables from .env file

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0) # Create a chatbot model