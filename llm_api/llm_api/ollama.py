from langchain_community.chat_models import ChatOllama


def new_ollama_instance() -> ChatOllama:
    """Create a new instance of the ChatOllama model"""
    # TODO: Centralize env var loading

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
    context_window = int(os.environ.get("CONTEXT_WINDOW", 8192))

    return ChatOllama(
        model="mistral:v0.3",
        temperature=0.3,
        base_url=ollama_base_url,
        num_ctx=context_window,
    )
