# Use the base image
FROM ollama/ollama

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

# Pull the mistral model
RUN ollama serve & \
	curl --retry 10 --retry-connrefused --retry-delay 1 http://localhost:11434/ && \
	curl -X POST -d '{"name": "mistral"}' http://localhost:11434/api/pull

# Copy the entrypoint script into the image
COPY ollama-docker-entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

HEALTHCHECK CMD curl --fail http://localhost:11434/api

# Run the mistral server
# CMD ["ollama", "serve", "&&", "ollama", "run", "mistral"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]