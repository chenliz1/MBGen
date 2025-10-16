#!/bin/bash
# ===========================================
# Vertex AI Runtime Post-Startup Script
# ===========================================

set -e  # Exit immediately if a command exits with a non-zero status

# 1. Go to working directory
cd /content || mkdir -p /content && cd /content

# 2. Upgrade pip and install Poetry
echo "Installing Poetry..."
pip install --upgrade pip
pip install poetry

# 3. Clone MBGen repository if not already present
if [ ! -d "MBGen" ]; then
    echo "Cloning MBGen repository..."
    git clone https://github.com/chenliz1/MBGen.git
else
    echo "MBGen repository already exists. Pulling latest changes..."
    cd MBGen && git pull && cd ..
fi

# 4. Install dependencies using Poetry
cd MBGen
echo "Installing dependencies into the system environment..."
poetry config virtualenvs.create false
poetry install --no-root

echo "Post-startup setup complete."
