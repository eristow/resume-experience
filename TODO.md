## TODO:
- [ ] Use same uuid in `context_manager` and `logging`

- [ ] Clean up code to not call `st.session_state` as much
  - Example: start of `user_input` to chatbot in `main.py`

 [ ] Split up the tuned Mistral model from the Streamlit container
  - 3 total containers: Streamlit, Ollama, Mistral
  - Streamlit will have to make API calls to Mistral
  - Mistral container will have a light API wrapper around the model

- [ ] For prod, probably just a VM with Docker and Docker Compose?
  - Make sure hot reload is turned off.


## EXTRA:
- [ ] Add actual concurrency for analysis?
  - Is this needed?

- [ ] Add a startup process for the Streamlit container so don't have to do an initial request to warm up the model
  - **Since analyze is now locked and will only perform one at a time, we won't need this.**
  - Potentially just load the model in the background


## DONE:
- [x] Fix tests, add new ones
  - [x] Comment out auto-resume buttons
  - [x] Then make a PR
- [x] Convert to using user-inputted job info, instead of having the LLM parse and do math
  - [x] Fix tests
  - [x] Add new tests for `components/job_input` and `logger`
  - [x] Filter out the word `description` from the Job Ad
  - [x] Figure out how to get it to stop thinking the Job Ad is relevant experience
    - `vectorstores` now only have relevant documents. But now having issues with loading model sometimes...
    - Creating `job_vectorstore`, only one document. Creating `resume_vectorstore`, both documents in both now.
    - I think it's because the embeddings are using the same class or something under-the-hood in langchain...
    - Something with the way the vectorstores/retrievers are being created. The `resume_retriever` should not have the Job Ad, and the `job_retriever` should not have the Job Info.
- [x] All vectorstores are being cleared when a new analysis happens
  - Error steps: if open chatbot then run new analysis in another session, open chatbot now uses context of another session
  - I think this should fix concurrent analysis as well. Test that...
  - Progress:
    - Does each session need separate embeddings? Similar to splitting up job/resume embeddings. Store this in `context_manager` so it can be cleared on each run?
    - Analysis, chat is working. But now documents are being combined... Check the `context_manager.vectorstores`
    - I think it's the `ContextManager`. The chatbot is not aware of a resume/job ad when `ContextManager.clear_context()` is called.
    - Investigate `vectorstore` usage. Since retrievers just hold a reference to the vectorstore.
    - Removed `AppState` from `session_state`. Now just using `session_state` directly.
    - Retrievers passed into `handle_chat()` are not correct. Seems like the retrievers are being overwritten after another session's analysis, but the rest of `AppState` is still correct.
  - [x] Create a unique session ID when initializing `session_state`. Use the session ID to index `ContextManager`, creating and clearing when needed for analysis runs.
- [x] Create test data objects, so I don't have to copy paste
  - [x] Add some dev button to make it a dev only feature
- [x] Test restricting the input of the resume. Only work experience
  - Potentially change UI to have manual input of work experience
    - Title, Company, Start date, End date, bullet points/description
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
