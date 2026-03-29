#!/bin/bash
set -e

echo "── Setting up project ──"

mkdir -p external
mkdir -p models/hf models/gguf

# Clone llama.cpp if not present

if [ ! -d "external/llama.cpp" ]; then
    echo "── Cloning llama.cpp ──"
    git clone https://github.com/ggerganov/llama.cpp.git external/llama.cpp
else
    echo "── llama.cpp already exists, skipping clone ──"
fi

# Build llama.cpp using CMake

echo "── Building llama.cpp ──"

cd external/llama.cpp

if [ ! -d "build" ]; then
    cmake -B build
fi

cmake --build build --config Release -j3

cd ../..

# Install Python dependencies

echo "── Installing Python dependencies ──"

pip install -r external/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

pip install \
langchain \
langchain-groq \
huggingface_hub \
python-dotenv

echo "✅ Setup complete."