import hashlib
import math
import re

from langchain_core.embeddings import Embeddings


class FeatureHashEmbeddings(Embeddings):
    """Small dependency-free embedding for local development, not production retrieval."""

    dimensions = 512

    @staticmethod
    def _embed(text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        vector = [0.0] * FeatureHashEmbeddings.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            index = number % FeatureHashEmbeddings.dimensions
            vector[index] += 1.0 if number & 1 else -1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
