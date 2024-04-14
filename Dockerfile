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


WORKDIR /app
COPY inputvideo.mp4 /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY process_video.py /app


ENTRYPOINT ["python", "process_video.py"]
