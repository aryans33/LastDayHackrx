#!/usr/bin/env bash
# build.sh - Render.com build script

set -o errexit  # Exit on error

echo "🚀 Starting build process..."

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev

# Upgrade pip and install build tools
echo "🔧 Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p vector_store

echo "✅ Build completed successfully!"
