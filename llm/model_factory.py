from typing import Literal, Optional
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

Provider = Literal["openai", "ollama"]

def get_chat_model(
    provider: Provider,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Returns a chat model. The agent will bind tools on top of this model.
    """
    provider = (provider or os.getenv("DEFAULT_PROVIDER", "ollama")).lower()

    if provider == "openai":
        model = model_name or os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "ollama":
        model = model_name or os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b")
        return ChatOllama(model=model, temperature=temperature, model_kwargs={"num_ctx": 4096})

    raise ValueError(f"Unknown provider: {provider}")