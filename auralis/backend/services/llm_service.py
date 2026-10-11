"""
LLM service - talks to Llama 3 through Ollama's streaming HTTP API.

Responsible for:
  - building the emotion-aware prompt (persona + emotion context + history)
  - streaming the response so we can measure Time-to-First-Token (TTFT)
  - keeping a bounded conversation history so prompts don't grow unbounded
"""

import requests
from backend.config import settings

SYSTEM_PERSONA = (
    "You are Auralis, a calm, emotionally aware conversational AI used in a "
    "crisis negotiation training simulator. Respond naturally and concisely "
    "(1-3 sentences), in a tone appropriate to the user's emotional state. "
    "Do not mention or explain the internal emotion classification - just "
    "respond the way a calm, empathetic human would."
)


class ConversationHistory:
    def __init__(self, max_turns: int = None):
        self.max_turns = max_turns or settings.LLM_MAX_HISTORY_TURNS
        self.turns: list[dict] = []  # [{"user": ..., "assistant": ...}, ...]

    def add(self, user_text: str, assistant_text: str):
        self.turns.append({"user": user_text, "assistant": assistant_text})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def as_prompt_block(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Auralis: {turn['assistant']}")
        return "\n".join(lines) + "\n"


def build_prompt(user_text: str, emotion: str, confidence: float, history: ConversationHistory) -> str:
    return (
        f"{SYSTEM_PERSONA}\n\n"
        f"User emotional state: {emotion} (confidence: {confidence})\n\n"
        f"Conversation so far:\n{history.as_prompt_block()}"
        f"User: {user_text}\n"
        f"Auralis:"
    )


def stream_llm_response(prompt: str, on_first_token=None):
    """
    Generator that streams tokens from Ollama and yields the full response
    text incrementally. Calls on_first_token() exactly once, the moment the
    first token is received (used for TTFT measurement).

    Yields str chunks. Caller concatenates them for the full response.
    """
    first_token_fired = False
    try:
        with requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": settings.LLM_MAX_RESPONSE_TOKENS},
            },
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                import json

                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    if not first_token_fired and on_first_token:
                        on_first_token()
                        first_token_fired = True
                    yield chunk
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "[Error: Could not reach Ollama. Is 'ollama serve' running?]"
    except Exception as exc:  # noqa: BLE001
        yield f"[Error calling LLM: {exc}]"
