FROM fedora:39

# Install Python and pip
RUN dnf install -y python3 python3-pip

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the entrypoint
CMD ["python3", "app.py"]