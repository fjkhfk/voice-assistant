import json
from pathlib import Path


class GreetingSystem:
    def __init__(self, greetings_path: str | Path | None = None) -> None:
        default_path: Path = Path(__file__).resolve().parents[1] / "config" / "greetings.json"
        self.greetings_path: Path = (
            Path(greetings_path) if greetings_path is not None else default_path
        )
        self.greetings: dict[str, str] = self._load_greetings()

    def _load_greetings(self) -> dict[str, str]:
        if self.greetings_path.exists():
            try:
                with self.greetings_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON format in {self.greetings_path}"
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to read {self.greetings_path}"
                ) from exc

            if not isinstance(data, dict):
                raise ValueError(
                    f"Expected a JSON object in {self.greetings_path}"
                )

            return {str(key): str(value) for key, value in data.items()}

        default_greetings: dict[str, str] = {
            "hello": "Hello! How can I help you?",
            "hi": "Hi! How can I help you?",
            "how are you": "I am doing well, thank you! How are you?",
            "bye": "Goodbye! Have a great day.",
            "goodbye": "Goodbye! Have a great day.",
        }

        try:
            self.greetings_path.parent.mkdir(parents=True, exist_ok=True)
            with self.greetings_path.open("w", encoding="utf-8") as file:
                json.dump(default_greetings, file, ensure_ascii=False, indent=4)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write {self.greetings_path}"
            ) from exc

        return default_greetings

    def respond_to_greeting(self, text: str) -> str | None:
        normalized_text: str = text.strip().lower()

        for trigger, response in self.greetings.items():
            if trigger.strip().lower() in normalized_text:
                return response

        return None