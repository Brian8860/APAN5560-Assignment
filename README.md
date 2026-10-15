# APAN 5560 - Assignment 1: Word Embedding API

## Overview

This project creates a simple Word Embedding API using FastAPI and spaCy.

The API takes a word as input and returns its embedding vector.

## Tools Used

- Python
- FastAPI
- spaCy
- Docker
- uv

## How to Run

### 1. Build the Docker Image

```bash
docker build -t apan5560-assignment1 .
```

### 2. Run the Docker Container

```bash
docker run --rm -p 8000:80 apan5560-assignment1
```

### 3. Open FastAPI

Go to:

http://127.0.0.1:8000/docs

## API Testing

### POST /embedding

Click "Try it out" and enter:

```json
{
  "word": "hello"
}
```

Click "Execute".

The API returns the input word and its embedding vector.

## Notes

- The API uses the spaCy `en_core_web_lg` language model.
- FastAPI handles the API requests.
- Docker is used to run the application.
