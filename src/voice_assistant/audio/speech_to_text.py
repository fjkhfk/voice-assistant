import logging
import speech_recognition as sr


class SpeechToText:
    def __init__(
        self,
        energy_threshold: int = 300,
        pause_threshold: float = 1.0,
        timeout: float = 8.0,
        phrase_time_limit: float = 15.0,
        language: str = "en-US",
    ) -> None:
        self.recognizer: sr.Recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.timeout: float = timeout
        self.phrase_time_limit: float = phrase_time_limit
        self.language: str = language

    def setup_microphone(self) -> None:
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def listen_once(self) -> str | None:
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
                text: str = self.recognizer.recognize_google(
                    audio, language=self.language
                )
                return text
            except sr.WaitTimeoutError:
                logging.debug("Listen timeout while waiting for user speech.")
                return None
            except sr.UnknownValueError:
                logging.debug("Could not understand user speech.")
                return None
            except sr.RequestError as exc:
                raise RuntimeError(
                    f"Speech recognition service error: {exc}"
                ) from exc