import asyncio
import logging
import os
import tempfile

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import edge_tts
import pygame

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(
        self,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+5%",
        volume: str = "+10%",
    ) -> None:
        self.voice: str = voice
        self.rate: str = rate
        self.volume: str = volume
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    async def generate_speech_file(self, text: str) -> str:
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file_path: str = temp_file.name

        await communicate.save(temp_file_path)
        return temp_file_path

    def play_audio_file(self, file_path: str) -> None:
        try:
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass

            pygame.mixer.music.load(file_path)
            pygame.mixer.music.set_volume(0.95)
            pygame.mixer.music.play()

            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(10)
        finally:
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass

            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return

        try:
            file_path: str = asyncio.run(self.generate_speech_file(text.strip()))
            self.play_audio_file(file_path)
        except Exception as exc:
            logger.error("TTS failed: %s", exc)