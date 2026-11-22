import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from voice_assistant.core.greeting_system import GreetingSystem


class HuggingFaceModel:
    def __init__(
        self,
        context: str = "You are a helpful voice assistant named Nova.",
        model_name: str = "deepset/roberta-base-squad2",
        env_var_name: str = "HUGGINGFACE_API_KEY",
        env_file: str | Path | None = None,
    ) -> None:
        if not context or not context.strip():
            raise ValueError("Context must be a non-empty string.")
        if not model_name or not model_name.strip():
            raise ValueError("Model name must be a non-empty string.")
        if not env_var_name or not env_var_name.strip():
            raise ValueError("Environment variable name must be a non-empty string.")

        self.context: str = context.strip()
        self.model_name: str = model_name.strip()
        self.env_var_name: str = env_var_name.strip()
        self.env_file: Path | None = Path(env_file) if env_file is not None else None

        self.api_key: str = ""
        self.api_key_loaded: bool = False
        self.is_ready: bool = False
        self._client: InferenceClient | None = None
        self._greeting_system: GreetingSystem = GreetingSystem()

        self._initialize_client()

    def _load_api_key(self) -> str:
        if self.env_file is not None:
            load_dotenv(dotenv_path=self.env_file)
        else:
            load_dotenv()

        token: str | None = os.getenv(self.env_var_name)
        if not token or not token.strip():
            raise ValueError(f"Missing or empty environment variable '{self.env_var_name}'.")

        return token.strip()

    def _initialize_client(self) -> None:
        try:
            self.api_key = self._load_api_key()
            self._client = InferenceClient(
                model=self.model_name,
                token=self.api_key,
            )
            self.api_key_loaded = True
            self.is_ready = True
        except Exception as exc:
            self.api_key_loaded = False
            self.is_ready = False
            logging.error("Failed to initialize Hugging Face client: %s", exc)
            raise RuntimeError(f"Failed to initialize Hugging Face client: {exc}") from exc

    def _extract_answer(self, response: object) -> tuple[str, float]:
        answer = getattr(response, "answer", None)
        score = getattr(response, "score", None)

        if not answer or not str(answer).strip():
            raise RuntimeError("Model returned an empty answer.")
        if score is None:
            raise RuntimeError("Model response did not include a confidence score.")

        return str(answer).strip(), float(score)

    def answer_question(self, question: str, context: str | None = None) -> tuple[str, float]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        resolved_context: str = self.context if context is None else context
        if not resolved_context or not resolved_context.strip():
            raise ValueError("Context cannot be empty.")

        if not self.is_ready or self._client is None:
            raise RuntimeError("AI model is not ready for inference.")

        cleaned_question: str = question.strip()
        cleaned_context: str = resolved_context.strip()

        greeting_reply: str | None = self._greeting_system.respond_to_greeting(cleaned_question)
        if greeting_reply:
            return greeting_reply, 1.0

        try:
            response = self._client.question_answering(
                question=cleaned_question,
                context=cleaned_context,
            )
            return self._extract_answer(response)
        except Exception as exc:
            logging.error("Failed to generate QA response via API: %s", exc)
            raise RuntimeError(f"Failed to generate QA response via API: {exc}") from exc

    def generate_reply(self, prompt: str) -> str:
        answer, _ = self.answer_question(question=prompt, context=self.context)
        return answer