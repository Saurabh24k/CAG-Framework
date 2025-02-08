# sentence_transformers_embedding.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .abstract_embedding import EmbeddingModel


class SentenceTransformersEmbeddingModel(EmbeddingModel):
    """Embedding model implementation using sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the SentenceTransformers embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                        Defaults to "all-MiniLM-L6-v2" for a balance of speed and quality.
        """
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name # Store model name for potential later use/info

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single input text.

        Args:
            text: Input text string.

        Returns:
            NumPy array representing the embedding vector for the input text.
        """
        return self.model.encode([text])[0]  # Encode returns a list of embeddings, we take the first one

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of input texts (batch embedding).

        Args:
            texts: List of input text strings.

        Returns:
            NumPy array where each row is the embedding vector for the corresponding text in the input list.
        """
        return self.model.encode(texts) # Returns a 2D numpy array

    def calculate_similarities(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate cosine similarities between a query embedding and a list of candidate embeddings.

        Uses cosine similarity from sklearn.metrics.pairwise.

        Args:
            query_embedding: NumPy array representing the embedding vector of the query.
            candidate_embeddings: List of NumPy arrays, where each array is an embedding vector for a candidate.

        Returns:
            NumPy array of cosine similarity scores. Each score represents the similarity
            between the query embedding and the corresponding candidate embedding.
        """
        # cosine_similarity expects 2D arrays, and returns a 2D array. We take the first row [0]
        return cosine_similarity([query_embedding], candidate_embeddings)[0]