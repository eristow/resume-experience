## TODO:
- [ ] Fix chatbot not having vectorstores from analysis
  - Due to clearing context_manager after analysis. Fix this

- [ ] First run can't be concurrent runs
  - Have to run only one at a time, then can run concurrently
  - I think because it has to load checkpoint shards, so maybe try downloading model in one file instead of shards

- [ ] Prevent user from interacting while processing
  - For extract text, analysis, and chatbot

- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] Test restricting the input of the resume. Only work experience
  - Potentially change UI to have manual input of work experience
    - Title, Company, Start date, End date, bullet points/description

- [ ] For prod, probably just a VM with Docker and Docker Compose?

## DONE:
- [x] Logger in `analysis.py` sometimes doesn't have UUID...
- [x] Test with concurrent analysis requests
  - I think Ollama is mixing the inputs. Debug this further
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
