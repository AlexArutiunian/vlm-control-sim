from typing import List, Dict
from openai import OpenAI
import settings

import base64, mimetypes, pathlib

def _file_to_data_uri(path: str) -> str:
    p = pathlib.Path(path)
    mime, _ = mimetypes.guess_type(p.name)
    if not mime:
        mime = "application/octet-stream"
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 60,
                 vision_model: str | None = None):             # <-- убедитесь, что есть этот аргумент
        self.client = OpenAI(api_key=api_key)
        self.model = model or settings.OPENAI_MODEL
        self.vision_model = vision_model or settings.OPENAI_VISION_MODEL
        self.timeout = timeout

    def chat(self, messages: list[dict], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
          #  max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
          #  temperature=kwargs.get("temperature", settings.LLM_TEMPERATURE),
          #  timeout=self.timeout,
        )
        return resp.choices[0].message.content

    # --- VLM: текст + картинка (для команд) ---
    def chat_vision(self, text: str, image_path: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": _file_to_data_uri(image_path)}},
                ],
            }
        ]
        resp = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
           # max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
           # temperature=kwargs.get("temperature", settings.LLM_TEMPERATURE),
           # timeout=self.timeout,
        )
        return resp.choices[0].message.content

    # --- VLM: безопасный бриф картинки (для вывода перед выполнением) ---
    def chat_vision_describe(self, image_path: str, **kwargs) -> str:
        sys = (
            "You are a careful, concise vision assistant for a robot controller. "
            "Do SHORT description only what is visible in the image that could affect motion planning: "
            "robot posture/pose cues, limbs orientation, obstacles, people, stairs, cables, markers, "
            "ground conditions, lighting, and any ambiguities. Include uncertainty estimates."
        )
        messages = [
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image for safe control context."},
                    {"type": "image_url", "image_url": {"url": _file_to_data_uri(image_path)}},
                ],
            },
        ]
        resp = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
          #  max_tokens=kwargs.get("max_tokens", 300),
          #  temperature=kwargs.get("temperature", 0.2),
          #  timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()
