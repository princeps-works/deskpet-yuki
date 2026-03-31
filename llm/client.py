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
        return response.choices[0].message.content or ""

    def multimodal_chat(
        self,
        *,
        user_content: list[dict],
        system_prompt: str,
        model_name: Optional[str] = None,
    ) -> str:
        if self._client is None:
            return ""

        selected_model = model_name or self.settings.vision_model_name
        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=False,
            )
        except Exception:
            return ""
        return response.choices[0].message.content or ""

    def describe_image(self, image: Image.Image, user_text: str, system_prompt: str) -> str:
        if self._client is None:
            return ""

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        image_url = f"data:image/png;base64,{b64}"

        return self.multimodal_chat(
            user_content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            system_prompt=system_prompt,
            model_name=self.settings.vision_model_name,
        )
