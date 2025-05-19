from pathlib import Path

# OpenAI Configuration
OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual API key
OPENAI_MODEL = "gpt-4"  # or "gpt-4-turbo-preview" depending on your needs
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# File paths and directories
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'
EXCEL_FOLDER = BASE_DIR / 'excel_outputs'
RAG_OUTPUTS = BASE_DIR / 'rag_outputs'

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
EXCEL_FOLDER.mkdir(exist_ok=True)
RAG_OUTPUTS.mkdir(exist_ok=True)

# Application settings
CHUNK_SIZE = 1000  # Large chunk size for better context

# Model settings
MAX_TOKENS = 4096
TEMPERATURE = 0.7 