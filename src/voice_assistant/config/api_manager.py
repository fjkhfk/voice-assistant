import os
from pathlib import Path

from dotenv import load_dotenv


class APIManager:
    OPENAI_PLACEHOLDERS = {"", "your_openai_api_key_here", "your-openai-api-key-here"}
    HF_PLACEHOLDERS = {"", "your_huggingface_api_key_here", "your-huggingface-api-key-here"}

    def __init__(self, env_file: str | Path | None = None) -> None:
        self.env_file: Path | None = Path(env_file) if env_file is not None else None
        self.load_api_keys()

    def load_api_keys(self) -> None:
        if self.env_file is not None:
            load_dotenv(dotenv_path=self.env_file)
        else:
            load_dotenv()

    def _validate_openai_key(self) -> None:
        openai_key: str | None = os.getenv("OPENAI_API_KEY")
        if openai_key is None or openai_key.strip() in self.OPENAI_PLACEHOLDERS:
            raise ValueError(
                "CRITICAL: OPENAI_API_KEY is missing or set to default placeholder."
            )

    def _validate_huggingface_key(self) -> None:
        hf_token: str | None = os.getenv("HUGGINGFACE_API_KEY")
        if hf_token is None or hf_token.strip() in self.HF_PLACEHOLDERS:
            raise ValueError(
                "CRITICAL: HUGGINGFACE_API_KEY is missing or set to default placeholder."
            )

    def check_api_keys(self, provider: str | None = None) -> None:
        if provider is None:
            self._validate_openai_key()
            self._validate_huggingface_key()
            return

        normalized = provider.strip().lower()
        if normalized == "openai":
            self._validate_openai_key()
        elif normalized == "huggingface":
            self._validate_huggingface_key()
        else:
            raise ValueError(f"Unsupported provider: {provider}")