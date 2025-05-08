# RAG Application using Amazon EC2

This repository contains a Retrieval-Augmented Generation (RAG) application built with FastAPI and deployed on Amazon EC2.

## Features

- PDF document upload and processing
- Vector search using Qdrant
- Question answering using LangChain and OpenAI
- Web UI for document management and querying

## Prerequisites

- Python 3.11 or higher
- Docker (for running Qdrant)
- OpenAI API key

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/junxng/Amazon-EC2-Ubuntu.git
# or
git clone git@github.com:junxng/Amazon-EC2-Ubuntu.git
```

```bash
cd Amazon-EC2-Ubuntu
```

### 2. Set up environment with `uv`

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

#### Install uv

You can install uv using curl (Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip:

```bash
pip install uv
```

#### Create and activate virtual environment

```bash
# Create a virtual environment
uv venv --python 3.11

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Set up environment variables

```bash
cp .env.example .env
nvim .env
# Or any of your own configuration like nano, vim, etc.
```

### 5. Start Qdrant (Vector Database)

The application uses Qdrant for storing vector embeddings. You need to run it using Docker:

```bash
docker pull qdrant/qdrant
```

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

This will start Qdrant on the default ports (6333 for HTTP and 6334 for GRPC).

### 6. Run the application

```bash
uv run fastapi run app/main.py
```

The application will be available at <http://127.0.0.1:8000>

## Usage

1. **Upload Documents**: Use the web UI to upload PDF files
2. **Ask Questions**: Type questions about the content of your documents
3. **Manage Documents**: View and delete documents in the knowledge base

## Development

- Project structure follows FastAPI best practices
- Dependencies are managed with pyproject.toml
- Code is organized into modules within the `app` directory

## Deployment on Amazon EC2

Instructions for deploying to Amazon EC2 can be found in the [deployment guide](docs/deployment.md).
