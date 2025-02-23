# Use a Python base image
FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev \
        curl \
        git \
        autoconf \
        automake \
        libtool \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Clone and build libpostal from source
RUN git clone https://github.com/openvenues/libpostal && \
    cd libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=/usr/local/data && \
    make -j4 && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf libpostal

# Set environment variables for python-libpostal
ENV CFLAGS="-I/usr/local/include"
ENV LDFLAGS="-L/usr/local/lib"

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Start the application
CMD ["python", "app.py"]