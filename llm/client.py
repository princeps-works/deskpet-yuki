from __future__ import annotations

import base64
import io
from typing import Optional

from PIL import Image

from desktop_pet.config.settings import Settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        if OpenAI is not None and settings.api_key:
            self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    @staticmethod
    def _extract_response_text(response) -> str:
        try:
            choices = getattr(response, "choices", None)
            if not choices:
                return ""
            first = choices[0]
            message = getattr(first, "message", None)
            if message is None:
                return ""

            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                    elif isinstance(item, str) and item.strip():
                        parts.append(item)
                return "\n".join(parts).strip()

            # Some providers may return non-standard fields while content is null.
            alt = getattr(message, "reasoning_content", None) or getattr(message, "refusal", None)
            if isinstance(alt, str):
                return alt
            return ""
        except Exception:
            return ""

    def chat(self, user_text: str, system_prompt: str) -> str:
        if self._client is None:
            return f"[离线回声] 你说的是: {user_text}"

        response = self._client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=False,
        )
        return self._extract_response_text(response)

    def multimodal_chat(
        self,
        *,
        user_content: list[dict],
        system_prompt: str,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> str:
        if self._client is None:
            return ""

        selected_model = model_name or self.settings.vision_model_name
        request_kwargs = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max(1, int(max_tokens))
        if timeout_sec is not None:
            request_kwargs["timeout"] = max(0.5, float(timeout_sec))
        response = self._client.chat.completions.create(**request_kwargs)
        return self._extract_response_text(response)

    def describe_image(self, image: Image.Image, user_text: str, system_prompt: str) -> str:
        if self._client is None:
            return ""

        buffer = io.BytesIO()
        prepared = image.convert("RGB")
        # JPEG significantly reduces payload size for screenshot uploads, improving latency/stability.
        prepared.save(buffer, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        image_url = f"data:image/jpeg;base64,{b64}"

        return self.multimodal_chat(
            user_content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            system_prompt=system_prompt,
            model_name=self.settings.vision_model_name,
            max_tokens=self.settings.mm_output_max_tokens,
            timeout_sec=self.settings.mm_timeout_sec,
        )
