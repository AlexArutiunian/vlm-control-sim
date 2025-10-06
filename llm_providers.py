import settings
from openai_llm import OpenAIClient

def build_llm() -> OpenAIClient:
    return OpenAIClient(
        api_key=settings.OPENAI_API_KEY or None,
        model=settings.OPENAI_MODEL,
        vision_model=settings.OPENAI_VISION_MODEL,
    )
