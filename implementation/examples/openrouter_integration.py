"""
OpenRouter Integration Example for Qwen3-VL-235B-A22B-Thinking.

Demonstrates how to call the OpenRouter chat completions endpoint with thinking
budget controls (reasoning tokens) for multimodal prompts.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests


ContentBlock = Dict[str, Any]
ReasoningOverride = Union[int, str, Dict[str, Any], None]


class Qwen3VLClient:
    """Client for interacting with Qwen3-VL Thinking via OpenRouter."""

    def __init__(
        self,
        api_key: str,
        *,
        model: Optional[str] = None,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        include_reasoning: Optional[bool] = None,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model or self._resolve_model_slug(
            os.environ.get("QWEN_OPENROUTER_MODEL_ID"),
            os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-235B-A22B-Thinking"),
        )
        self.reasoning_max_tokens = reasoning_max_tokens
        if self.reasoning_max_tokens is None:
            self.reasoning_max_tokens = self._int_from_env(
                "OPENROUTER_REASONING_MAX_TOKENS"
            )
        self.reasoning_effort = reasoning_effort or self._effort_from_env(
            os.environ.get("OPENROUTER_REASONING_EFFORT")
        )
        self.reasoning_exclude = self._bool_from_env(
            os.environ.get("OPENROUTER_REASONING_EXCLUDE", "false")
        )
        if include_reasoning is None:
            include_reasoning = self._bool_from_env(
                os.environ.get("OPENROUTER_INCLUDE_REASONING", "true")
            )
        self.include_reasoning = include_reasoning
        self.referer = os.environ.get("OPENROUTER_SITE_URL")
        self.app_name = os.environ.get("OPENROUTER_APP_NAME")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        *,
        thinking_budget: ReasoningOverride = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Send a text-only chat prompt."""

        blocks: List[ContentBlock] = [{"type": "text", "text": prompt}]
        return self._make_request(blocks, thinking_budget, temperature, max_tokens)

    def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        *,
        thinking_budget: ReasoningOverride = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Analyze a single image with an optional thinking budget."""

        blocks = [self._image_block(image_path), {"type": "text", "text": prompt}]
        return self._make_request(blocks, thinking_budget, temperature, max_tokens)

    def analyze_document(
        self,
        document_path: Union[str, Path, bytes],
        task: str,
        *,
        thinking_budget: ReasoningOverride = None,
    ) -> Dict[str, Any]:
        prompt = f"Please {task} from this document."
        return self.analyze_image(
            document_path,
            prompt,
            thinking_budget=thinking_budget,
        )

    def process_video(
        self,
        video_frames: Sequence[Union[str, Path, bytes]],
        prompt: str,
        *,
        thinking_budget: ReasoningOverride = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Send a sequence of frames for reasoning about a video."""

        blocks: List[ContentBlock] = [self._image_block(frame) for frame in video_frames]
        blocks.append({"type": "text", "text": prompt})
        return self._make_request(blocks, thinking_budget, temperature, max_tokens)

    # ------------------------------------------------------------------
    # Low-level request builder
    # ------------------------------------------------------------------
    def _make_request(
        self,
        content_blocks: List[ContentBlock],
        thinking_budget: ReasoningOverride,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        reasoning_payload, include_reasoning = self._reasoning_payload(thinking_budget)
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload
            if include_reasoning:
                payload["include_reasoning"] = True

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_name:
            headers["X-Title"] = self.app_name

        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        return {
            "text": message.get("content"),
            "reasoning": message.get("reasoning") or message.get("reasoning_details"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "usage": usage,
            "raw": data,
        }

    # ------------------------------------------------------------------
    # Support utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_slug(
        explicit: Optional[str],
        fallback: str,
    ) -> str:
        if explicit:
            return explicit
        if "/" in fallback:
            namespace, model_name = fallback.split("/", 1)
            return f"{namespace.lower()}/{model_name.lower()}"
        return fallback.lower()

    @staticmethod
    def _int_from_env(name: str) -> Optional[int]:
        value = os.environ.get(name)
        if not value:
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer.") from exc

    @staticmethod
    def _bool_from_env(value: Optional[str]) -> bool:
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _effort_from_env(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        effort = value.strip().lower()
        if effort not in {"low", "medium", "high"}:
            raise ValueError("Reasoning effort must be one of: low, medium, high.")
        return effort

    def _reasoning_payload(
        self,
        override: ReasoningOverride,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        include_reasoning = self.include_reasoning and not self.reasoning_exclude
        payload: Dict[str, Any] = {}

        effective_max = self.reasoning_max_tokens
        effective_effort = self.reasoning_effort
        effective_exclude = self.reasoning_exclude

        if isinstance(override, int):
            effective_max = override
            effective_effort = None
        elif isinstance(override, str):
            effective_effort = self._effort_from_env(override)
            effective_max = None
        elif isinstance(override, dict):
            payload.update(override)
            effective_max = payload.get("max_tokens", effective_max)
            effective_effort = payload.get("effort", effective_effort)
            if "exclude" in payload:
                effective_exclude = bool(payload["exclude"])

        if effective_max is not None:
            payload["max_tokens"] = effective_max
            effective_effort = None
        if effective_effort:
            if "max_tokens" in payload:
                raise ValueError("Specify either a max reasoning token budget or an effort level, not both.")
            payload["effort"] = effective_effort
        if effective_exclude:
            payload["exclude"] = True
            include_reasoning = False

        return (payload or None, include_reasoning)

    @staticmethod
    def _image_block(image_data: Union[str, Path, bytes]) -> ContentBlock:
        encoded, media_type = Qwen3VLClient._encode_image(image_data)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{encoded}"
            },
        }

    @staticmethod
    def _encode_image(image_data: Union[str, Path, bytes]) -> Tuple[str, str]:
        media_type = "image/png"
        if isinstance(image_data, bytes):
            raw = image_data
        else:
            path = Path(image_data)
            if path.exists():
                raw = path.read_bytes()
                media_type = path.suffix.lower().lstrip(".")
                media_type = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp",
                    "bmp": "image/bmp",
                }.get(media_type, "image/png")
            else:
                # Assume already base64 encoded string
                if isinstance(image_data, str):
                    if image_data.startswith("data:"):
                        header, encoded = image_data.split(",", 1)
                        media_type = header.split(";")[0].split(":")[1]
                        return encoded, media_type
                    return image_data, media_type
                raise FileNotFoundError(f"Image path {image_data} does not exist.")

        encoded = base64.b64encode(raw).decode("utf-8")
        return encoded, media_type


def main() -> None:
    """Demonstrate thinking-budget aware calls."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY before running the example.")

    client = Qwen3VLClient(api_key)

    # Example 1: Text-only reasoning with explicit token budget
    text_response = client.chat(
        "Outline a research plan for classifying rare plant species from field photos.",
        thinking_budget=3000,
    )
    print("\nText Response:\n", text_response["text"])
    print("Reasoning tokens used:", text_response["reasoning_tokens"])

    # Example 2: Image reasoning with high-effort thinking
    image_path = Path("path/to/document_page.png")
    if image_path.exists():
        image_response = client.analyze_image(
            image_path,
            "Summarize the key findings in this document and capture numerical tables.",
            thinking_budget="high",
        )
        print("\nImage Analysis:\n", image_response["text"])
        if image_response["reasoning"]:
            print("Reasoning snippet:", image_response["reasoning"]) 

    # Example 3: Video frames with custom reasoning payload
    frame_paths = [Path("path/to/frame1.jpg"), Path("path/to/frame2.jpg"), Path("path/to/frame3.jpg")]
    existing_frames = [frame for frame in frame_paths if frame.exists()]
    if existing_frames:
        video_response = client.process_video(
            existing_frames,
            "Explain the sequence of events shown in these frames.",
            thinking_budget={"max_tokens": 4096, "exclude": False},
        )
        print("\nVideo Reasoning:\n", video_response["text"])
        if video_response["reasoning"]:
            print("First reasoning block:", video_response["reasoning"])


if __name__ == "__main__":
    main()
