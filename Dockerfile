# Use an official TensorFlow Docker image as the base
FROM tensorflow/tensorflow:2.15.0-gpu

# Upgrade pip
RUN python -m pip install --upgrade pip
RUN python3 -m pip install wheel

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the local code to the container image
# COPY . /app
COPY lambda_function.py /app
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the Lambda handler command
CMD ["python", "lambda_function.py"]
