#!/bin/sh
/bin/ollama serve &

# Wait for the server to start
sleep 5

# Start mistral model
/bin/ollama run mistral &

# Wait indefinitely to keep the container running
tail -f /dev/null