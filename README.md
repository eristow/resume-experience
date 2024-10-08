# resume-experience

Compare a job description to a resume and extract the number of years of relevant work experience from the resume.

## PRE-REQs

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

## RUNNING THE APP

- Start Ollama:

  ```bash
  ollama serve
  ```

- Run the app:

  ```bash
  streamlit run app.py
  ```
