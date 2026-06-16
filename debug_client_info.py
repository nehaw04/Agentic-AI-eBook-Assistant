import inspect
from google import genai

print('genai version:', getattr(genai, '__version__', 'unknown'))
print('Client class methods:')
for name in sorted([n for n in dir(genai.Client) if not n.startswith('_')]):
    print(name)
