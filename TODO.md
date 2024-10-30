## TODO:
- [ ] Remove docker GHA workflow after it completes once.

- [ ] Test with concurrent analysis requests

- [ ] Create state_manager class for Streamlit state
- [ ] Add configuration management for hardcoded values
- [ ] Improve error handling and logging
- [ ] Separate UI components from business logic in main.py
- [ ] Enhance context_manager class
- [ ] Optimize the custom_embeddings class

- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] For prod, probably just a VM with Docker and Docker Compose?
