## TODO:
- [ ] Test with concurrent analysis requests
  - I think Ollama is mixing the inputs. Debug this further

- [ ] Ensure vectorstores for other runs are not being cleared for current run
  - I think this is happening because the vectorstore is a class variable instead of an instance variable

- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] Add a startup process for the Streamlit container so don't have to do an initial request to warm up the model
  - Potentially just load the model in the background

- [ ] Test restricting the input of the resume. Only work experience
  - Potentially change UI to have manual input of work experience
    - Title, Company, Start date, End date, bullet points/description

- [ ] For prod, probably just a VM with Docker and Docker Compose?
  - Make sure hot reload is turned off.

## DONE:
- [x] Chatbot Error: `TypeError: Expected a Runnable, callable or dict.Instead got an unsupported type: <class 'str'>`
- [x] Logging only working for main file
  - Move `setup_logging` to separate file and use that instead
- [x] Improve error handling and logging
  - Add UUID for logging during concurrent runs
    - https://docs.python.org/3/howto/logging-cookbook.html#adding-contextual-information-to-your-logging-output
- [x] Fix docker-compose build/up
- [x] Fix "Install python dependencies" step of GHA streamlit test workflow
  - Ensure docker build and run still work
- [x] Add tests for new files
- [x] Separate UI components from business logic in main.py
- [x] Add typing to everything
- [x] Add configuration management for hardcoded values
- [x] Create state_manager class for Streamlit state
