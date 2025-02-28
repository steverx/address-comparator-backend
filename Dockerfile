# Use a consistent name for the builder stage
FROM steverx/libpostal-builder:latest as libpostal-builder

# --- Final Backend Image ---
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
        gosu \
        libpq-dev \
        curl \
        grep \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Copy libpostal files from the builder stage
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/include/libpostal/ /usr/local/include/libpostal/
COPY --from=libpostal-builder --chown=appuser:appuser /usr/local/data/ /usr/local/data/

# Set environment variables
ENV LIBPOSTAL_INCLUDE_DIR=/usr/local/include
ENV LIBPOSTAL_LIB_DIR=/usr/local/lib
ENV LIBPOSTAL_DATA_DIR=/usr/local/data
ENV LD_LIBRARY_PATH="${LIBPOSTAL_LIB_DIR}:${LD_LIBRARY_PATH}"
ENV PORT=5000
ENV FLASK_APP=app.py

# Run ldconfig to update shared libraries
RUN ldconfig

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN grep -v "postal" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements_filtered.txt

# Install Flask
RUN pip install --no-cache-dir flask requests

# Add to your Dockerfile
RUN pip install --no-cache-dir scikit-learn pandas numpy usaddress fuzzywuzzy python-Levenshtein jellyfish pycountry

# Add to your Dockerfile
RUN pip install --no-cache-dir gunicorn

# Add to your Dockerfile
RUN pip install --no-cache-dir waitress

# Copy application code
COPY --chown=appuser:appuser . .

# Ensure directories exist
RUN mkdir -p /app/api /app/config /app/utils /app/tasks /app/tests && \
    touch /app/api/__init__.py /app/config/__init__.py /app/utils/__init__.py /app/tasks/__init__.py /app/tests/__init__.py

# Create a launcher script that uses Gunicorn
RUN echo '#!/bin/bash\nset -e\necho "Starting nginx..."\nnginx -g "daemon on;"\necho "Starting Gunicorn..."\ncd /app\nexec gunicorn --bind "0.0.0.0:${PORT}" "app:create_app()"' > /app/start.sh && \
    chmod +x /app/start.sh

# Command to run the app
CMD ["python", "-m", "waitress", "--port=5000", "--call", "app:create_app"]