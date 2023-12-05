
FROM python:3.10

# Set the working directory to /server
WORKDIR /app

# Copy requirements.txt to /server
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy to the Container
COPY . .

# Run Command
CMD uvicorn server.main:app --reload --port 8000 --host 0.0.0.0