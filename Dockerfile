# Use a Python base image
FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        python3-dev \
        libpq-dev \
        curl \
        git \
        autoconf \
        automake \
        libtool \
        pkg-config \
        build-essential \
        libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

# Create data directory
RUN mkdir -p /usr/local/data

# Clone and build libpostal from source with verbose output
RUN git clone https://github.com/openvenues/libpostal && \
    cd libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=/usr/local/data --prefix=/usr/local && \
    make CFLAGS="-O3 -fPIC" -j4 && \
    make install && \
    ldconfig

# Set environment variables for python-libpostal
ENV CFLAGS="-I/usr/local/include"
ENV LDFLAGS="-L/usr/local/lib"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

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