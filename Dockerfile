
FROM python:3.10

# Set the working directory to /server
WORKDIR /server

# Copy ./requirements/prod.txt to /server
COPY ./requirements/prod.txt /server/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /server/requirements.txt

# Copy to the Container
COPY /app /server/app

# Run Command
CMD uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
