#!/usr/bin/env bash
# build.sh - Simplified Render.com build script

set -o errexit  # Exit on error

echo "🚀 Starting simplified build process..."

# Upgrade pip and install build tools first
echo "🔧 Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies with pre-compiled wheels only
echo "📚 Installing Python dependencies..."
pip install --only-binary=all --no-cache-dir -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p vector_store

echo "✅ Build completed successfully!"
