#!/usr/bin/env bash
# start.sh - Render.com startup script

set -o errexit  # Exit on error

echo "🚀 Starting HackRX Document Q&A API..."

# Set environment variables
export PYTHONPATH="/opt/render/project/src:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p vector_store

# Start the FastAPI application
echo "🌐 Starting FastAPI server..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
