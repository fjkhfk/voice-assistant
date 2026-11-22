import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from voice_assistant.audio.speech_to_text import SpeechToText
from voice_assistant.audio.text_to_speech import TextToSpeech
from voice_assistant.audio.wake_word import DEFAULT_WAKE_WORDS, WakewordDetector
from voice_assistant.config.api_manager import APIManager


class ChatModel(Protocol):
    def generate_reply(self, prompt: str) -> str:
        ...


class VoiceAssistantPipeline:
    EXIT_TRIGGERS: tuple[str, ...] = (
        "exit",
        "quit",
        "stop",
        "goodbye",
        "bye",
        "sleep",
        "shut down",
    )

    def __init__(
        self,
        model: ChatModel,
        env_file: str | Path | None = None,
        wake_words: Iterable[str] = DEFAULT_WAKE_WORDS,
        voice: str = "en-GB-RyanNeural",
        language: str = "en-US",
    ) -> None:
        if model is None:
            raise ValueError("Model instance cannot be None.")

        self._api_manager: APIManager = APIManager(env_file=env_file)
        self._model: ChatModel = model
        self._wake_word_detector: WakewordDetector = WakewordDetector(
            wake_words=wake_words,
            language=language,
        )
        self._speech_to_text: SpeechToText = SpeechToText(language=language)
        self._text_to_speech: TextToSpeech = TextToSpeech(voice=voice)
        self._session_start: float = 0.0
        self._question_count: int = 0

        if getattr(self._model, "api_key_loaded", True) is False:
            raise RuntimeError("Model API key is not loaded.")

    def _print_turn(speaker: str, message: str) -> None:
        print(f"{speaker}: {message}")

    def _configure_microphones(self) -> None:
        try:
            self._speech_to_text.setup_microphone()
            self._wake_word_detector.setup_microphone()
        except Exception as exc:
            raise RuntimeError(f"Failed to configure microphones: {exc}") from exc

    def _should_exit(self, user_text: str) -> bool:
        normalized: str = " ".join(user_text.strip().lower().split())
        pattern: str = (
            r"\b(?:" + "|".join(re.escape(w) for w in self.EXIT_TRIGGERS) + r")\b"
        )
        return bool(re.search(pattern, normalized))

    def _process_turn(self, prompt: str) -> dict[str, Any]:
        cleaned_prompt: str = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty.")

        start_time: float = time.perf_counter()
        reply: str = self._model.generate_reply(cleaned_prompt)
        self._print_turn("Nova", reply)

        self._text_to_speech.speak(reply)
        elapsed_seconds: float = time.perf_counter() - start_time
        self._question_count += 1

        return {
            "question_number": self._question_count,
            "prompt": cleaned_prompt,
            "reply": reply,
            "elapsed_seconds": elapsed_seconds,
        }

    def get_session_summary(self) -> dict[str, float | int]:
        if self._session_start <= 0.0:
            raise RuntimeError("Session has not been started.")

        total_seconds: float = max(0.0, time.perf_counter() - self._session_start)
        average_seconds: float = (
            total_seconds / self._question_count if self._question_count > 0 else 0.0
        )

        return {
            "question_count": self._question_count,
            "total_minutes": total_seconds / 60.0,
            "average_seconds_per_question": average_seconds,
        }

    def run(self) -> None:
        self._configure_microphones()
        self._session_start = time.perf_counter()
        self._question_count = 0

        while True:
            detected: bool = self._wake_word_detector.wait_for_wakeword()
            if not detected:
                continue

            greeting: str = "Yes, I am listening."
            self._print_turn("Nova", greeting)
            self._text_to_speech.speak(greeting)
            time.sleep(0.2)

            should_shutdown: bool = False
            while not should_shutdown:
                user_prompt: str | None = self._speech_to_text.listen_once()
                if not user_prompt or not user_prompt.strip():
                    continue

                cleaned_user_prompt: str = user_prompt.strip()
                self._print_turn("You", cleaned_user_prompt)

                if self._should_exit(cleaned_user_prompt):
                    farewell: str = "Goodbye. Have a great day."
                    self._print_turn("Nova", farewell)
                    self._text_to_speech.speak(farewell)
                    should_shutdown = True
                    break

                try:
                    self._process_turn(cleaned_user_prompt)
                except Exception as exc:
                    self._print_turn("Nova", f"Something went wrong: {exc}")

            if should_shutdown:
                break