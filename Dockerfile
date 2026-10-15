FROM python:3.13-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /code

# Install Python dependencies
COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen --no-dev

# Download spaCy English language model
RUN uv run --no-sync python -m spacy download en_core_web_lg

# Copy application files
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./models /code/models

# Expose FastAPI port
EXPOSE 80

# Start FastAPI
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]