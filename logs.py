import os, logging
from dotenv import load_dotenv
from langchain import hub

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
WATCH_DIR= os.path.join(ROOT_PATH,"data/docs/")
VECTORSTORE_DIR = os.path.join(ROOT_PATH,"data/vectorstore/")
META_FILE = os.path.join(VECTORSTORE_DIR,"meta.json")
SLEEP_TIME = 100  # Seconds
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PROMPT = hub.pull("rlm/rag-prompt")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FAITHFULNESS_THRESHOLD = 0.6
ANSWER_RELEVANCY_THRESHOLD = 0.7
CONTEXT_RECALL_THRESHOLD = 0.8