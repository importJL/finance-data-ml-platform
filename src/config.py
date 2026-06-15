import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ALPHAVANTAGE_API_KEY not found. "
        "Create a .env file in the project root with:\n"
        "ALPHAVANTAGE_API_KEY=your_key_here"
    )
