from pathlib import Path
p = Path(r"d:/The_Internship_sem/agentic-ai-rag/.venv/Lib/site-packages/pinecone/__init__.py")
content = '''"""Pinecone SDK package initializer.

This file exposes the main client classes so downstream packages
can import `from pinecone import Pinecone` and
`from pinecone import PineconeAsyncio`.

Note: editing site-packages is intended as a temporary developer
workaround to allow the application to run in this environment.
"""

from .pinecone import Pinecone  # noqa: E402
from .pinecone_asyncio import PineconeAsyncio  # noqa: E402

__all__ = ["Pinecone", "PineconeAsyncio"]
'''
try:
    p.write_text(content, encoding='utf-8')
    print('patched', p)
except Exception as e:
    print('error', e)
