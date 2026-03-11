import base64
import io
import mimetypes
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


def _clean_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nil"}:
        return None
    return text


class OpenAICompatVLM:
    def __init__(self, model_path, args):
        self.model = model_path
        self.api_key = _clean_optional(getattr(args, "test_api_key", None)) or _clean_optional(os.environ.get("TEST_API_KEY")) or _clean_optional(os.environ.get("OPENAI_API_KEY"))
        self.base_url = _clean_optional(getattr(args, "test_base_url", None)) or _clean_optional(os.environ.get("TEST_BASE_URL")) or _clean_optional(os.environ.get("OPENAI_BASE_URL"))
        if not self.api_key:
            raise ValueError("Closed-source API model requires --test_api_key (or TEST_API_KEY / OPENAI_API_KEY).")
        if not self.base_url:
            raise ValueError("Closed-source API model requires --test_base_url (or TEST_BASE_URL / OPENAI_BASE_URL).")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError("openai package is required for OpenAI-Compatible API inference.") from exc

        self._client_cls = OpenAI
        self._thread_local = threading.local()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.max_workers = max(1, int(getattr(args, "test_max_workers", 8)))
        self.max_retries = max(1, int(getattr(args, "test_max_retries", 8)))
        self.retry_delay_base = max(0.1, float(getattr(args, "test_retry_delay_base", 2.0)))
        self.image_detail = getattr(args, "test_image_detail", "auto")
        self.request_timeout = float(getattr(args, "test_timeout", 600))

    def process_messages(self, messages):
        return messages

    def _get_client(self):
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = self._client_cls(
                api_key=self.api_key,
                base_url=self.base_url.rstrip("/"),
                timeout=self.request_timeout,
            )
            self._thread_local.client = client
        return client

    def _load_pil_image(self, image):
        pil_image = None
        source_path = None
        if isinstance(image, dict):
            if image.get("bytes"):
                pil_image = Image.open(io.BytesIO(image["bytes"]))
                source_path = image.get("path")
            elif image.get("path"):
                source_path = image["path"]
                pil_image = Image.open(source_path)
            else:
                for key in ("image", "pil_image", "img"):
                    if isinstance(image.get(key), Image.Image):
                        pil_image = image[key]
                        break
                if pil_image is None:
                    for value in image.values():
                        if isinstance(value, Image.Image):
                            pil_image = value
                            break
        elif isinstance(image, str):
            source_path = image
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if pil_image is None:
            raise ValueError("Failed to parse image input.")
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return pil_image, source_path

    def _encode_image(self, image) -> Tuple[str, str]:
        pil_image, source_path = self._load_pil_image(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        mime_type = "image/jpeg"
        if source_path:
            guessed = mimetypes.guess_type(source_path)[0]
            if guessed and guessed.startswith("image/"):
                mime_type = guessed
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), mime_type

    def _iter_images(self, messages: Dict[str, Any]) -> Iterable[Any]:
        if messages.get("image") is not None:
            yield messages["image"]
        if messages.get("images"):
            for image in messages["images"]:
                yield image

    def _build_user_content(self, messages):
        content: List[Dict[str, Any]] = []
        prompt = messages.get("prompt") or messages.get("text") or ""

        for image in self._iter_images(messages):
            image_b64, mime_type = self._encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}",
                    "detail": self.image_detail,
                },
            })

        if prompt:
            content.append({"type": "text", "text": prompt})
        if not content:
            content = [{"type": "text", "text": ""}]
        return content

    def _build_messages(self, messages):
        if isinstance(messages, str):
            messages = {"prompt": messages}
        api_messages = []
        system_text = messages.get("system")
        if system_text:
            api_messages.append({"role": "system", "content": system_text})
        api_messages.append({"role": "user", "content": self._build_user_content(messages)})
        return api_messages

    def _extract_text(self, response) -> str:
        try:
            message = response.choices[0].message
        except Exception as exc:
            raise RuntimeError(f"Unexpected API response format: {response}") from exc

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if text:
                        texts.append(text)
                else:
                    text = getattr(block, "text", None)
                    if text:
                        texts.append(text)
            if texts:
                return "\n".join(texts)

        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if reasoning:
            return str(reasoning)
        return ""

    def _single_request(self, messages):
        client = self._get_client()
        api_messages = self._build_messages(messages)
        request_kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
        }
        response = client.chat.completions.create(**request_kwargs)
        return self._extract_text(response)

    def generate_output(self, messages):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._single_request(messages)
            except Exception as exc:
                last_error = exc
                sleep_s = self.retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"[OpenAICompatVLM] request failed (attempt {attempt}/{self.max_retries}): {exc}")
                if attempt < self.max_retries:
                    time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI-Compatible API inference failed after {self.max_retries} retries: {last_error}")

    def generate_outputs(self, messages_list):
        if not messages_list:
            return []
        results = [None] * len(messages_list)
        worker_count = min(self.max_workers, len(messages_list))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self.generate_output, messages): idx
                for idx, messages in enumerate(messages_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
