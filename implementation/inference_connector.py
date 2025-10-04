import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


class InferenceConnector:
    """Helper for routing multimodal prompts to different inference providers."""

    def __init__(self) -> None:
        # Shared configuration
        self.qwen_model_id = os.environ.get(
            "QWEN_VL_MODEL_ID",
            "Qwen/Qwen3-VL-235B-A22B-Thinking",
        )

        # Hugging Face
        self.hf_api_key = os.environ.get("HF_API_KEY")

        # OpenRouter
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_endpoint = os.environ.get(
            "OPENROUTER_ENDPOINT",
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.openrouter_site_url = os.environ.get("OPENROUTER_SITE_URL")
        self.openrouter_app_name = os.environ.get("OPENROUTER_APP_NAME")
        self._openrouter_model_id = self._resolve_openrouter_model_id()
        self._openrouter_reasoning_max_tokens = self._int_from_env(
            "OPENROUTER_REASONING_MAX_TOKENS"
        )
        self._openrouter_reasoning_effort = self._effort_from_env(
            os.environ.get("OPENROUTER_REASONING_EFFORT")
        )
        self._openrouter_reasoning_exclude = self._bool_from_env(
            os.environ.get("OPENROUTER_REASONING_EXCLUDE", "false")
        )
        self._openrouter_include_reasoning = self._bool_from_env(
            os.environ.get("OPENROUTER_INCLUDE_REASONING", "true")
        )

        # Hyperbolic
        self.hyperbolic_api_key = os.environ.get("HYPERBOLIC_API_KEY")
        self.hyperbolic_endpoint = os.environ.get("HYPERBOLIC_ENDPOINT")

    # ------------------------------------------------------------------
    # Provider helpers
    # ------------------------------------------------------------------
    def infer_huggingface(
        self,
        prompt: str,
        image_data: Optional[Union[str, bytes, Path]] = None,
        max_new_tokens: int = 512,
    ) -> Union[str, List[str]]:
        """Run inference using the local Hugging Face transformers stack.

        Notes:
            Qwen3-VL-235B-A22B-Thinking is extremely large (MoE 235B). Running
            this locally requires multiple high-memory GPUs. Prefer a hosted
            inference endpoint unless you have sufficient hardware.
        """

        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen3VLMoeForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.57.0.dev0 and torch are required for Qwen3-VL support."
            ) from exc

        processor = AutoProcessor.from_pretrained(
            self.qwen_model_id,
            token=self.hf_api_key,
        )
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.qwen_model_id,
            device_map="auto",
            torch_dtype=getattr(torch, "bfloat16", torch.float32),
            attn_implementation="flash_attention_2",
            token=self.hf_api_key,
        )

        messages = self._build_transformers_messages(prompt, image_data)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_length = inputs["input_ids"].shape[1]
        trimmed = generated[:, prompt_length:]
        output_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if len(output_text) == 1 else output_text

    def infer_openrouter(
        self,
        prompt: str,
        image_data: Optional[Union[str, bytes, Path]] = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Send a request to OpenRouter with optional thinking budget."""

        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not set in environment.")

        content_blocks = self._build_openrouter_content_blocks(prompt, image_data)
        payload: Dict[str, Any] = {
            "model": self._openrouter_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        reasoning_payload = self._build_reasoning_payload()
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload
            if (
                self._openrouter_include_reasoning
                and not reasoning_payload.get("exclude", False)
            ):
                payload["include_reasoning"] = True

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-Title"] = self.openrouter_app_name

        response = requests.post(
            self.openrouter_endpoint,
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return {
            "content": message.get("content"),
            "reasoning": message.get("reasoning")
            or message.get("reasoning_details"),
            "raw": data,
        }

    def infer_hyperbolic(
        self,
        prompt: str,
        image_data: Optional[Union[str, bytes, Path]] = None,
    ) -> Dict[str, Any]:
        if not self.hyperbolic_endpoint or not self.hyperbolic_api_key:
            raise ValueError("Hyperbolic endpoint or API key not set in environment.")

        payload: Dict[str, Any] = {"inputs": prompt}
        if image_data:
            encoded, _ = self._ensure_base64_image(image_data)
            payload["image"] = encoded

        headers = {"Authorization": f"Bearer {self.hyperbolic_api_key}"}
        response = requests.post(
            self.hyperbolic_endpoint,
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def infer(
        self,
        provider: str,
        prompt: str,
        image_data: Optional[Union[str, bytes, Path]] = None,
        **kwargs: Any,
    ) -> Any:
        provider = provider.lower()
        if provider in {"huggingface", "hf"}:
            return self.infer_huggingface(prompt, image_data, **kwargs)
        if provider == "openrouter":
            return self.infer_openrouter(prompt, image_data, **kwargs)
        if provider == "hyperbolic":
            return self.infer_hyperbolic(prompt, image_data)
        raise ValueError(f"Unsupported provider: {provider}")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _int_from_env(name: str) -> Optional[int]:
        value = os.environ.get(name)
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer.") from exc
        return parsed

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
            raise ValueError(
                "OPENROUTER_REASONING_EFFORT must be one of: low, medium, high."
            )
        return effort

    def _build_reasoning_payload(self) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {}
        if self._openrouter_reasoning_max_tokens is not None:
            payload["max_tokens"] = self._openrouter_reasoning_max_tokens
        if self._openrouter_reasoning_effort:
            if payload:
                raise ValueError(
                    "Specify either OPENROUTER_REASONING_MAX_TOKENS or "
                    "OPENROUTER_REASONING_EFFORT, not both."
                )
            payload["effort"] = self._openrouter_reasoning_effort
        if self._openrouter_reasoning_exclude:
            payload["exclude"] = True
        return payload or None

    def _build_openrouter_content_blocks(
        self,
        prompt: str,
        image_data: Optional[Union[str, bytes, Path]],
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        if image_data is not None:
            encoded, media_type = self._ensure_base64_image(image_data)
            blocks.append(
                {
                    "type": "input_image",
                    "image": {
                        "data": encoded,
                        "media_type": media_type or "image/png",
                    },
                }
            )
        blocks.append({"type": "input_text", "text": prompt})
        return blocks

    def _build_transformers_messages(
        self,
        prompt: str,
        image_data: Optional[Union[str, bytes, Path]],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if image_data is not None:
            image_input = self._prepare_transformers_image(image_data)
            content.append({"type": "image", "image": image_input})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_transformers_image(
        self,
        image_data: Union[str, bytes, Path],
    ) -> Union[str, Any]:
        from io import BytesIO
        from PIL import Image

        if isinstance(image_data, Path):
            image_bytes = image_data.read_bytes()
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        elif isinstance(image_data, str):
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            elif os.path.exists(image_data):
                image_bytes = Path(image_data).read_bytes()
            else:
                # Assume this is a URL or Hugging Face can fetch it directly
                return image_data
        else:
            raise TypeError("Unsupported image type for transformers input.")

        return Image.open(BytesIO(image_bytes))

    def _ensure_base64_image(
        self,
        image_data: Union[str, bytes, Path],
    ) -> Tuple[str, Optional[str]]:
        media_type = "image/png"
        if isinstance(image_data, Path):
            raw = image_data.read_bytes()
            media_type = mimetypes.guess_type(str(image_data))[0] or media_type
        elif isinstance(image_data, bytes):
            raw = image_data
        elif isinstance(image_data, str):
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
                return encoded, media_type
            path = Path(image_data)
            if path.exists():
                raw = path.read_bytes()
                media_type = mimetypes.guess_type(str(path))[0] or media_type
            else:
                # Assume already base64 encoded
                return image_data, media_type
        else:
            raise TypeError("Unsupported image format. Provide bytes, base64, or a file path.")

        encoded = base64.b64encode(raw).decode("utf-8")
        return encoded, media_type

    def _resolve_openrouter_model_id(self) -> str:
        explicit = os.environ.get("QWEN_OPENROUTER_MODEL_ID")
        if explicit:
            return explicit
        if "/" in self.qwen_model_id:
            namespace, model_name = self.qwen_model_id.split("/", 1)
            return f"{namespace.lower()}/{model_name.lower()}"
        return self.qwen_model_id.lower()


# Example usage:
# connector = InferenceConnector()
# response = connector.infer("openrouter", "Explain the image", image_path)
# print(response["content"])
