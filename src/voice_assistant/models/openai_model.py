import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

from voice_assistant.core.greeting_system import GreetingSystem


class OpenAIModel:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        env_var_name: str = "OPENAI_API_KEY",
        env_file: str | Path | None = None,
        greetings_path: str | Path | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        default_greetings_path: Path = (
            Path(__file__).resolve().parents[1] / "config" / "greetings.json"
        )
        self.greetings_path: Path = (
            Path(greetings_path) if greetings_path is not None else default_greetings_path
        )
        if not model_name or not model_name.strip():
            raise ValueError("Model name must be a non-empty string.")
        if not env_var_name or not env_var_name.strip():
            raise ValueError("Environment variable name must be a non-empty string.")

        self.model_name: str = model_name.strip()
        self.env_var_name: str = env_var_name.strip()
        self.env_file: Path | None = Path(env_file) if env_file is not None else None
        self.system_prompt: str = system_prompt
        self.greetings: GreetingSystem = GreetingSystem(greetings_path)
        self._client: OpenAI | None = None
        self.api_key_loaded: bool = False
        self._load_api_key()

    def _load_api_key(self) -> None:
        try:
            if self.env_file is not None:
                load_dotenv(dotenv_path=self.env_file)
            else:
                load_dotenv()

            raw_key: str | None = os.getenv(self.env_var_name)
            api_key: str = raw_key.strip() if raw_key else ""
            if (
                not api_key
                or api_key.startswith("your-openai-api-key-here")
                or api_key == "your_openai_api_key_here"
            ):
                raise ValueError(f"Missing or invalid API key in '{self.env_var_name}'.")

            self._client = OpenAI(api_key=api_key)
            self.api_key_loaded = True
        except Exception as exc:
            self.api_key_loaded = False
            self._client = None
            logging.error("Failed to load OpenAI configuration: %s", exc)
            raise RuntimeError(f"Failed to initialize OpenAI client: {exc}") from exc

    def generate_reply(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not self.api_key_loaded or self._client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        cleaned_prompt: str = prompt.strip()

        greeting_reply: str | None = self.greetings.respond_to_greeting(cleaned_prompt)
        if greeting_reply:
            return greeting_reply

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": cleaned_prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )

            raw_content: str | None = response.choices[0].message.content
            reply: str = raw_content.strip() if raw_content else ""
            if not reply:
                raise RuntimeError("Model returned an empty reply.")

            return reply
        except RateLimitError as exc:
            raise RuntimeError("OpenAI rate limit exceeded.") from exc
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI authentication failed. Check API key.") from exc
        except APIConnectionError as exc:
            raise RuntimeError("Network error while contacting OpenAI API.") from exc
        except Exception as exc:
            logging.error("Unexpected LLM error: %s", exc)
            raise RuntimeError(f"Failed to generate response via OpenAI API: {exc}") from exc