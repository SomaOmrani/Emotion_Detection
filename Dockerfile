# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    alsa-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Ensure Git LFS is installed
RUN apt-get update && apt-get install -y git-lfs
RUN git lfs install

# Pull the large files with Git LFS
RUN git lfs pull

# Expose the port the app runs on
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "streamlit_app_with_dash.py"]
