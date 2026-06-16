import os
root=r'D:\The_Internship_sem\agentic-ai-rag\.venv\Lib\site-packages\pinecone'
found=False
for dirpath,dirs,files in os.walk(root):
    for f in files:
        if f.endswith('.py'):
            path=os.path.join(dirpath,f)
            try:
                with open(path,'r',encoding='utf-8') as fh:
                    txt=fh.read()
            except Exception:
                continue
            if 'class Pinecone' in txt or 'class PineconeAsyncio' in txt or 'class PineconeClient' in txt or "class Pinecone" in txt:
                print(path)
                found=True
if not found:
    print('no matches')
