# 1. Use a lightweight Python 3.11 base
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only requirements first (to leverage Docker caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project code
COPY . .

# 6. Expose the ports for FastAPI (8000) and Gradio (7860)
EXPOSE 8000
EXPOSE 7860

# 7. Start the Backend API and the UI Frontend together
# This command runs the FastAPI server in the background and the UI in the foreground
CMD python -m src.main & python src/ui.py


