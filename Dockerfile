# Build stage
FROM python:3.9-slim as builder

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
    && rm -rf /var/lib/apt/lists/*

# Create data directory
RUN mkdir -p /usr/local/data

# Clone and build libpostal with progress output
RUN git clone --depth 1 https://github.com/openvenues/libpostal && \
    cd libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=/usr/local/data --prefix=/usr/local && \
    make CFLAGS="-O2 -fPIC" -j$(nproc) V=1 && \
    make install && \
    ldconfig

# Final stage
FROM python:3.9-slim

# Copy libpostal from builder
COPY --from=builder /usr/local/lib/libpostal.* /usr/local/lib/
COPY --from=builder /usr/local/include/libpostal /usr/local/include/libpostal
COPY --from=builder /usr/local/data /usr/local/data

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgcc1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/* && \
    ldconfig

# Set environment variables
ENV CFLAGS="-I/usr/local/include"
ENV LDFLAGS="-L/usr/local/lib"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Start application
CMD ["python", "app.py"]