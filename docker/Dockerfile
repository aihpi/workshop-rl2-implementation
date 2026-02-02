# Start from official Python 3.11 image (slim = minimal size, ~150MB)
FROM python:3.11-slim

# Install curl (needed to download uv)
# --no-install-recommends: only install curl, not optional extras
# rm -rf ...: delete package cache to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure environment:
# - Add uv to PATH so we can run it
# - UV_SYSTEM_PYTHON=1: use the container's Python, don't create a virtual environment
ENV PATH="/root/.local/bin:$PATH" UV_SYSTEM_PYTHON=1

# Set working directory (where our code will live)
WORKDIR /workspace

# Copy dependency list into the image
COPY pyproject.toml .

# Install all dependencies listed in pyproject.toml
RUN uv sync

# Default command when container starts
CMD ["bash"]
