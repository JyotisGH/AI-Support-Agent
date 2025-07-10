import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from logs import OPENAI_API_KEY,logger

load_dotenv()


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

if not OPENAI_API_KEY:
    logger.info("OpenAI_API_KEY is not available in environment")
    raise EnvironmentError("OpenAI_API_KEY is not available in environment")

def get_llm() -> ChatOpenAI:
    """
    Initializes and returns a ChatOpenAI model instance.

    Returns:
        ChatOpenAI: An instance of ChatOpenAI with the specified configuration.
    """
    logger.info("Initializing ChatOpenAI model with gpt-4")
    return ChatOpenAI(model="gpt-4",temperature=0, api_key=OPENAI_API_KEY)

# def get_obs_path(extention:str) -> str:
#     """
#     This function will take the extended ural and create obs path
#     """
#     return os.path.abspath(os.path.join(os.path.dirname(__file__), extention))

def get_embadings():
    """
    This function will return embadings model
    """
    logger.info("Initializing OpenAIEmbeddings.")
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)