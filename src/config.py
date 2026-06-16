import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.massive.com"

if not API_KEY:
    raise RuntimeError(
        "ALPHAVANTAGE_API_KEY not found. "
        "Create a .env file in the project root with:\n"
        "ALPHAVANTAGE_API_KEY=your_key_here"
    )
