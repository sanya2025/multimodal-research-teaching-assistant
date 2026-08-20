"""Tests for VLMClient.generate() and VLMClient.caption()."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

from mrta.core.exceptions import LLMError
from mrta.multimodal.vlm_client import VLMClient

_WHITE = Image.new("RGB", (4, 4), color=(255, 255, 255))
_MOCK_RESPONSE = {"message": {"content": "This is a mock response."}}


def _make_client(model: str = "qwen2.5vl:latest") -> VLMClient:
    return VLMClient(model=model)


class TestGenerate:
    def test_returns_string(self) -> None:
        client = _make_client()
        with patch("mrta.multimodal.vlm_client.ollama.chat", return_value=_MOCK_RESPONSE):
            result = client.generate("Describe this.", images=[_WHITE])
        assert isinstance(result, str)
        assert result == "This is a mock response."

    def test_passes_prompt_and_images(self) -> None:
        client = _make_client()
        with patch("mrta.multimodal.vlm_client.ollama.chat") as mock_chat:
            mock_chat.return_value = _MOCK_RESPONSE
            client.generate("My prompt", images=[_WHITE])
        call_args = mock_chat.call_args
        msg = call_args.kwargs["messages"][0]
        assert msg["content"] == "My prompt"
        assert len(msg["images"]) == 1  # one base64-encoded image

    def test_multiple_images(self) -> None:
        client = _make_client()
        with patch("mrta.multimodal.vlm_client.ollama.chat") as mock_chat:
            mock_chat.return_value = _MOCK_RESPONSE
            client.generate("Compare these.", images=[_WHITE, _WHITE])
        msg = mock_chat.call_args.kwargs["messages"][0]
        assert len(msg["images"]) == 2

    def test_raises_llm_error_on_404(self) -> None:
        import ollama as ollama_lib

        client = _make_client()
        err = ollama_lib.ResponseError("not found")
        err.status_code = 404
        with patch("mrta.multimodal.vlm_client.ollama.chat", side_effect=err):
            with pytest.raises(LLMError, match="not installed"):
                client.generate("prompt", images=[_WHITE])

    def test_raises_llm_error_on_generic_exception(self) -> None:
        client = _make_client()
        with patch("mrta.multimodal.vlm_client.ollama.chat", side_effect=RuntimeError("boom")):
            with pytest.raises(LLMError):
                client.generate("prompt", images=[_WHITE])

    def test_model_property(self) -> None:
        client = VLMClient(model="mymodel:latest")
        assert client.model == "mymodel:latest"


class TestCaption:
    def test_caption_delegates_to_generate(self) -> None:
        client = _make_client()
        with patch("mrta.multimodal.vlm_client.ollama.chat", return_value=_MOCK_RESPONSE):
            result = client.caption(_WHITE)
        assert result == "This is a mock response."

    def test_caption_uses_default_prompt(self) -> None:
        client = _make_client()
        with patch("mrta.multimodal.vlm_client.ollama.chat") as mock_chat:
            mock_chat.return_value = _MOCK_RESPONSE
            client.caption(_WHITE)
        msg = mock_chat.call_args.kwargs["messages"][0]
        assert "graduate student" in msg["content"].lower()

    def test_caption_accepts_custom_prompt(self) -> None:
        client = _make_client()
        with patch("mrta.multimodal.vlm_client.ollama.chat") as mock_chat:
            mock_chat.return_value = _MOCK_RESPONSE
            client.caption(_WHITE, prompt="What colour is this?")
        msg = mock_chat.call_args.kwargs["messages"][0]
        assert "colour" in msg["content"]


class TestIsAvailable:
    def test_returns_true_when_model_found(self) -> None:
        with patch("mrta.multimodal.vlm_client.ollama.show", return_value={}):
            assert VLMClient.is_available("qwen2.5vl:latest") is True

    def test_returns_false_when_model_not_found(self) -> None:
        with patch("mrta.multimodal.vlm_client.ollama.show", side_effect=Exception("not found")):
            assert VLMClient.is_available("nonexistent:model") is False
