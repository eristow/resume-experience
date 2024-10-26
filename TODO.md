## TODO:
- [ ] Fix GHA docker-test failing due to no disk space
	- Try self-hosting to get a bigger disk?
	-	Self-host on one of my laptops instead of a cloud service

- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] For prod, probably just a VM with Docker and Docker Compose?
