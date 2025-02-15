# Start with a Python base image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY b.py /app

# Install FastAPI and httpx (and uvicorn if needed)
RUN pip install fastapi httpx uvicorn

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn

# Expose the port FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "b:app", "--host", "0.0.0.0", "--port", "8000"]
