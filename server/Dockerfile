# Use the official Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app will run on
EXPOSE 8090

# Set the default command to run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
