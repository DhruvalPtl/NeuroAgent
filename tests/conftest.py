# Load environment variables from .env file for tests
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
