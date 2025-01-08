# resume-experience

Compare a job description to a resume and extract the number of years of relevant work experience from the resume.


## RUN APP WITH DOCKER
### PRE-REQS
- Docker:

  - https://docs.docker.com/get-docker/

- Docker Compose:

  - https://docs.docker.com/compose/install/

### RUNNING THE APP
- Build the app with Docker Compose:
  
    ```bash
    docker compose build
    ```

- Start the app with Docker Compose:

  ```bash
  docker compose -p "resume-experience" up
  ```

- Build the Docker image:

  ```bash
  docker build -t resume_experience_streamlit -f streamlit.Dockerfile .
  docker build -t resume_experience_ollama -f ollama.Dockerfile .
  ```

- Start an individual container:
  ```bash
  docker run --gpus all -p 8501:8501 -v ${PWD}/src/models:/models -v ${PWD}/src:/src:ro resume_experience_streamlit
  docker run --gpus all -p 11434:11434 resume_experience_ollama
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

  - Install from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### RUNNING THE APP

- Start Ollama:

  ```bash
  ollama serve
  ```

- Run the app:

  ```bash
  streamlit run main.py
  ```

### TESTING
- Run the tests:

  ```bash
  cd src
  python -m pytest
  ```
