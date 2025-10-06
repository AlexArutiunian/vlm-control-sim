import os
from secrets import OPENAI_KEY
# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_KEY)
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5")
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.2"))
PROMPT_PATH      = os.getenv("PROMPT_PATH", "prompt.txt")
