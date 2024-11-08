#!/bin/sh
/bin/ollama serve &

# Wait for the server to be ready
until curl -s http://localhost:11434/api/tags >/dev/null; do
    echo "Waiting for Ollama server to be ready..."
    sleep 2
done

echo "Ollama server is ready. Verifying model..."

# Verify model exists
if ! curl -s http://localhost:11434/api/tags | grep -q "mistral:v0.3"; then
    echo "Error: mistral:v0.3 model not found in image"
    exit 1
fi

echo "Starting mistral:v0.3 model..."
/bin/ollama run mistral:v0.3 &

# Wait indefinitely to keep the container running
tail -f /dev/null