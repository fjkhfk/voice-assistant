from collections.abc import Iterable
import speech_recognition as sr

DEFAULT_WAKE_WORDS: tuple[str, ...] = (
    "wake up nova",
    "hey nova",
    "nova",
)

class WakewordDetector:

    def __init__(
        self,
        energy_threshold: int = 300,
        pause_threshold: float = 1.0,
        timeout: float = 5.0,
        phrase_time_limit: float = 4.0,
        wake_words: Iterable[str] = DEFAULT_WAKE_WORDS,
        language: str = "en-US",
    ) -> None:
        self.recognizer: sr.Recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.timeout: float = timeout
        self.phrase_time_limit: float = phrase_time_limit
        self.wake_words: tuple[str, ...] = self._normalize_wake_words(wake_words)
        self.language: str = language

    def _normalize_wake_words(self, wake_words: Iterable[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for word in wake_words:
            if not isinstance(word, str):
                raise TypeError("Each wake word must be a string.")
            normalized: str = " ".join(word.strip().lower().split())
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)

        if not cleaned:
            raise ValueError("At least one non-empty wake word must be provided.")

        return tuple(sorted(cleaned, key=len, reverse=True))

    def _matches_wake_word(self, transcript: str) -> str | None:
        for word in self.wake_words:
            if word in transcript:
                return word
        return None

    def _print_turn(self, speaker: str, message: str) -> None:
        print(f"{speaker}: {message}")

    def setup_microphone(self) -> None:
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        self._print_turn("Nova", "Listening for wake word.")

    def wait_for_wakeword(self) -> bool:
        with sr.Microphone() as source:
            while True:
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_time_limit,
                    )
                    text: str = self.recognizer.recognize_google(
                        audio, language=self.language
                    ).lower()
                    self._print_turn("You", text)

                    if self._matches_wake_word(text) is not None:
                        return True
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as exc:
                    raise RuntimeError(
                        f"Speech recognition service error: {exc}"
                    ) from exc