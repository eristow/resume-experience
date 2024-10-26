## TODO:
- [ ] Add GHA for building images and running containers
- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model
- [ ] For prod, probably just a VM with Docker and Docker Compose?
