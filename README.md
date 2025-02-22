# resume-experience

Compare a job ad to a resume and extract the number of years of relevant work experience from the resume.


## RUN APP WITH DOCKER
### PRE-REQS
- Docker:

  - https://docs.docker.com/get-docker/

- Docker Compose:

  - https://docs.docker.com/compose/install/

- A Hugging Face account with an access token.

- To download the model from HuggingFace into the Streamlit Docker container using `src/scripts/docker_download_model.py`:
  - Ensure the file `src/scripts/model_token.py` exists and has the following content:
  ```python
  ACCESS_TOKEN = "<YOUR_HUGGING_FACE_ACCESS_TOKEN>"
  ```

### RUNNING THE APP WITH DOCKER COMPOSE
- Build the app with Docker Compose:
  
    ```bash
    docker compose build
    ```

- Start the app with Docker Compose:

  ```bash
  docker compose -p "resume-experience" up
  ```

### RUNNING THE APP WITH ONLY DOCKER
- Build the Docker image:

  ```bash
  docker build -t resume_experience_streamlit -f streamlit.Dockerfile .
  docker build -t resume_experience_ollama -f ollama.Dockerfile .
  docker build -t resume_experiende_llm_api -f llm_api.Dockerfile .
  ```

# TODO: Adjust docker run for streamlit container. Probably won't need `--gpus all` after adding `llm_api`, or volume to `models/`.
- Start an individual container:
  ```bash
  docker run --gpus all -p 8501:8501 -v ${PWD}/src/models:/models -v ${PWD}/src:/src:ro resume_experience_streamlit
  docker run --gpus all -p 11434:11434 resume_experience_ollama
  docker run --user uwsgi --gpus all -p 80:80 -p 5000:5000 -v /home/eristow/projects/resume-experience/src/models:/models -v /home/eristow/projects/resume-experience/llm_api:/llm_api:ro resume_experience_llm_api
  ```


## RUN APP LOCALLY:
### PRE-REQs

- Ollama:

  - https://ollama.com
  - https://python.langchain.com/v0.2/docs/integrations/chat/ollama/
    - Setup section
  ```bash
  ollama pull mistral:v0.3
  ```

- poppler:

  - https://poppler.freedesktop.org/

  ```bash
  sudo apt install -y poppler-utils
  ```

- tesseract:

  - https://tesseract-ocr.github.io/tessdoc/Installation.html

  ```bash
  sudo apt install tesseract-ocr
  ```

- Mistral:

  - Run `download_model.py`:

    ```bash
    python download_model.py
    ```
  
  - Ensure the mistral model is in the `src/models` directory:

    ```bash
    ls src/models
    ```

- Python packages:

  - Install from `requirements.txt` files:
    - This can be done in two separate python venv's.

    ```bash
    pip install -r requirements.txt
    pip install -r llm_api/requirements.txt
    ```

### RUNNING THE APP

- Start Ollama:

  ```bash
  ollama serve
  ```

- Run the app:

  ```bash
  <ACTIVATE streamlit VIRTUAL ENV>
  streamlit run main.py
  ```

- Run the llm_api:
  
  - Production mode:

    ```bash
    cd llm_api
    <ACTIVATE llm_api VIRTUAL ENV>
    uwsgi --http 0.0.0.0:5000 --module wsgi:app --master
    ```
  
  - Development mode:

    ```bash
    cd llm_api
    export FLASK_ENV=development
    <ACTIVATE llm_api VIRTUAL ENV>
    flask --app llm_api run --debug
    ```

### TESTING
- Run the tests:

  ```bash
  cd src
  python -m pytest
  ```
