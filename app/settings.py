import os
import sys
from pathlib import Path

# Runtime Python files live in app/, while local secrets stay in the repository
# root and are intentionally excluded by .gitignore.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from secrets_keys import OPENAI_KEY
except ImportError:
    OPENAI_KEY = None

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
PROMPT_PATH = os.getenv("PROMPT_PATH", str(REPO_ROOT / "prompt.txt"))
