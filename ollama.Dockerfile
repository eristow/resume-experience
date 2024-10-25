# Use the base image
FROM ollama/ollama

LABEL maintainer="eristow"
LABEL version="1.0"
LABEL description="Ollama Dockerfile with Mistral for resume-experience app"

# Set up the necessary environment variables and configurations
ENV NVIDIA_VISIBLE_DEVICES=all

# Define the volume for storing Ollama data
VOLUME /root/.ollama

# Expose the port
EXPOSE 11434

# Update apt and install curl
RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive \
	apt-get install --no-install-recommends --assume-yes \
	curl

# Start server, wait for it, and pull model during build
RUN ollama serve & \
	timeout 60s bash -c 'until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done' && \
	ollama pull mistral:v0.3 && \
	pkill ollama

# Copy the entrypoint script into the image
COPY ollama-docker-entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

HEALTHCHECK CMD curl --fail http://localhost:11434/api

# Run the mistral server
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]