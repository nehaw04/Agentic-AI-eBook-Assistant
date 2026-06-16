import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

# Use the installed Google GenAI client to list available models.
try:
    client = genai.Client()
    models = client.models.list()
    print('MODELS:')
    for m in models[:50]:
        print(m)
except Exception as e:
    import traceback
    traceback.print_exc()