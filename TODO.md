## TODO:
- [ ] Remove docker GHA workflow after it completes once.

- [ ] Fix input_truncated response from ollama
  - Seems to happen on the second and subsequent requests
    - Seems like there is a second output of "Chunk 0 token count:" on the second run. Investigate this.

- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] For prod, probably just a VM with Docker and Docker Compose?
