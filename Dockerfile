# Use the official Python 3.12 image
FROM python:3.12

# Upgrade pip
RUN python -m pip install --upgrade pip


# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA's package repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# Install TensorRT packages
RUN apt-get update && apt-get install -y \
    libnvinfer8 \
    libnvinfer-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the local code to the container image
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the Lambda handler command
CMD ["python", "lambda_function.py"]
