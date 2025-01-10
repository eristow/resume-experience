## TODO:
- [ ] Convert to using user-inputted job info, instead of having the LLM parse and do math
  - [ ] Fix tests
  - [ ] Make a PR after this is done

- [ ] All vectorstores are being cleared when a new analysis happens
  - Error steps: if open chatbot then run new analysis in another session, open chatbot now uses context of another session
  - I think this should fix concurrent analysis as well. Test that...
  - Progress:
    - Analysis, chat is working. But now documents are being combined... Check the `context_manager.vectorstores`
    - I think it's the `ContextManager`. The chatbot is not aware of a resume/job description when `ContextManager.clear_context()` is called.
    - Investigate `vectorstore` usage. Since retrievers just hold a reference to the vectorstore.
    - Removed `AppState` from `session_state`. Now just using `session_state` directly.
    - Retrievers passed into `handle_chat()` are not correct. Seems like the retrievers are being overwritten after another session's analysis, but the rest of `AppState` is still correct.
  - [x] Create a unique session ID when initializing `session_state`. Use the session ID to index `ContextManager`, creating and clearing when needed for analysis runs.

- [ ] Clean up code to not call `st.session_state` as much
  - Example: start of `user_input` to chatbot in `main.py`

- [ ] Figure out when to clean up old sessions in `context_manager`

- [ ] Add a startup process for the Streamlit container so don't have to do an initial request to warm up the model
  - Potentially just load the model in the background

- [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] Add actual concurrency for analysis?
  - Is this needed?
  - Separation of vectorstores is a pre-req

- [ ] Test restricting the input of the resume. Only work experience
  - Potentially change UI to have manual input of work experience
    - Title, Company, Start date, End date, bullet points/description

- [ ] For prod, probably just a VM with Docker and Docker Compose?
  - Make sure hot reload is turned off.

## DONE:
- [x] Use a lock for analysis requests
  - Lock isn't being released after analysis run
- [x] Test with concurrent analysis requests
  - I think Ollama is mixing the inputs. Debug this further
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
