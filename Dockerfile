FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly copy model file
COPY catboost_model.cbm .

# Copy the rest of the application
COPY . .

# Create input and output directories
RUN mkdir -p input output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"] 