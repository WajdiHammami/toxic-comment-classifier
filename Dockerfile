# Base image with python
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . .

# Set default env vars
ENV MODEL_PATH=/app/artifacts/distilbert_model_2.pth
ENV THRESHOLDS_PATH=/app/artifacts/thresholds.npy
ENV DEVICE=cpu


# EXPOSE the port the app runs on
EXPOSE 8000


# RUN FASTAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]