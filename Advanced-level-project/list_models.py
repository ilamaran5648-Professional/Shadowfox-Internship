import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

client = genai.Client(api_key=api_key)

print("=== Available Gemini Models (generateContent supported) ===")
for m in client.models.list():
    name = m.name
    if any(kw in name.lower() for kw in ["flash", "pro", "gemini"]):
        print(f"  {name}")
