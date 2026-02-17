from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
print(f"Key loaded: {'Yes' if key else 'No'}")
if key:
    print(f"Key length: {len(key)}")
    print(f"First 10 chars: {key[:10]}")
    print(f"Last 4 chars: {key[-4:]}")
    print(f"Contains whitespace: {any(c.isspace() for c in key)}")
    print(f"Representation: {repr(key)}")
