import asyncio
from dotenv import load_dotenv
from src.main import ask_question, Query

load_dotenv()

async def main():
    try:
        result = await ask_question(Query(question="test question"))
        print(result)
    except Exception:
        import traceback
        traceback.print_exc()

asyncio.run(main())
