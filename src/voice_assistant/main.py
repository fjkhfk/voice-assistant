import argparse

from voice_assistant.config.api_manager import APIManager
from voice_assistant.core.pipeline import VoiceAssistantPipeline
from voice_assistant.models.huggingface_model import HuggingFaceModel
from voice_assistant.models.openai_model import OpenAIModel

AssistantModel = OpenAIModel | HuggingFaceModel


def _prompt_hf_context() -> str:
    user_context = input("Enter reference context for Hugging Face QA: ").strip()
    if not user_context:
        raise ValueError("Reference context cannot be empty.")
    return user_context


def _build_model(provider: str) -> AssistantModel:
    if provider == "openai":
        return OpenAIModel()
    if provider == "huggingface":
        context = _prompt_hf_context()
        return HuggingFaceModel(context=context)
    raise ValueError(f"Unsupported provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the voice assistant.")
    parser.add_argument(
        "--provider",
        choices=("openai", "huggingface"),
        default="openai",
        help="LLM provider to use for answering questions.",
    )
    args = parser.parse_args()

    APIManager().check_api_keys(provider=args.provider)
    model = _build_model(args.provider)
    pipeline = VoiceAssistantPipeline(model=model)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nNova: Shutting down.")


if __name__ == "__main__":
    main()