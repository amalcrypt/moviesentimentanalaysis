# Use a fast, official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the specifically slimmed API requirements file
COPY api-requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r api-requirements.txt

# Copy source, API code, and the locally trained DistilBERT models
COPY api/ api/
COPY src/ src/
COPY models/ models/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the Fast API server via Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
