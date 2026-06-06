"""mrta.multimodal — CLIP image embeddings and VLM captioning."""

from mrta.multimodal.clip_embedder import CLIPEmbedder
from mrta.multimodal.vlm_client import VLMClient

__all__ = ["CLIPEmbedder", "VLMClient"]
