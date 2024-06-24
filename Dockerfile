# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
