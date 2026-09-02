
"""
A4 - LLM provider abstraction.

Retrieval must stay open-source (A2 requirement), but GENERATION may use a hosted API.
This interface lets the provider be swapped purely via env var, so the assignment can be
evaluated with Ollama (no API key needed) even if the submitter's own dev loop used Groq.
"""

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class OllamaProvider(LLMProvider):
    """Fully local, no API key. Requires `ollama serve` running and a model pulled,
    e.g. `ollama pull llama3.2:1b`.
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        import requests

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                "stream": False,

                # Keep the small local model focused and deterministic.
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=120,
        )

        resp.raise_for_status()

        return resp.json()["message"]["content"]


class GroqProvider(LLMProvider):
    """Hosted, fast, free tier available. Needs GROQ_API_KEY."""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None
    ):
        self.model = model
        self.api_key = api_key or os.environ["GROQ_API_KEY"]

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        import requests

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                "temperature": 0.1,
            },
            timeout=60,
        )

        resp.raise_for_status()

        return resp.json()["choices"][0]["message"]["content"]


class MockProvider(LLMProvider):
    """Used only for testing the citation-guard/threshold logic without a real LLM call
    (e.g. in CI, or in an offline sandbox). Returns a canned response with a deliberately
    fabricated citation, so tests can prove the validator actually strips it.
    """

    def __init__(self, canned_response: str):
        self.canned_response = canned_response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        return self.canned_response


def get_llm_provider() -> LLMProvider:
    """Selects provider purely from env var -- this is the "sit behind an interface,
    swappable by env var" requirement from the brief.
    """

    provider_name = os.environ.get(
        "LLM_PROVIDER",
        "ollama"
    ).lower()

    if provider_name == "ollama":

        return OllamaProvider(
            model=os.environ.get(
                "OLLAMA_MODEL",
                "llama3.2:1b"
            )
        )

    elif provider_name == "groq":

        return GroqProvider(
            model=os.environ.get(
                "GROQ_MODEL",
                "llama-3.1-8b-instant"
            )
        )

    else:

        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider_name}"
        )


