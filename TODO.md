## TODO:
- [ ] Fix "Install python dependencies" step of GHA streamlit test workflow
  - Ensure docker build and run still work
- [ ] Add tests for new files

- [ ] Test with concurrent analysis requests

- Suggestions from Claude:
  - [ ] Enhance context_manager class
  - [ ] Optimize the custom_embeddings class
  - [ ] Improve error handling and logging

- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] For prod, probably just a VM with Docker and Docker Compose?

## DONE:
- [x] Separate UI components from business logic in main.py
- [x] Add typing to everything
- [x] Add configuration management for hardcoded values
- [x] Create state_manager class for Streamlit state
