# openai_embedding.py
from typing import List
import numpy as np
from openai import OpenAI
from openai import APIError  # Import OpenAI specific error
from sklearn.metrics.pairwise import cosine_similarity

from .abstract_embedding import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model implementation using OpenAI's API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize the OpenAI client for embedding generation.

        Args:
            api_key: OpenAI API key. Must be provided.
            model: Name of the OpenAI embedding model to use.
                   Defaults to "text-embedding-3-small", a good balance of cost and performance.
        Raises:
            ValueError: if api_key is not provided.
        """
        if not api_key:
            raise ValueError("OpenAI API key must be provided to use OpenAIEmbeddingModel.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._model_name = model # Store model name for potential use


    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single input text using OpenAI's API.

        Args:
            text: Input text string.

        Returns:
            NumPy array representing the embedding vector from OpenAI API.
        Raises:
            APIError: If there's an issue with the OpenAI API request.
        """
        try:
            response = self.client.embeddings.create(model=self.model, input=[text])
            return np.array(response.data[0].embedding)
        except APIError as e:
            print(f"[OpenAIEmbeddingModel] API Error during embedding generation: {e}") # Log error with model context
            raise # Re-raise the exception for higher-level handling


    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of input texts using OpenAI's batch API.

        Args:
            texts: List of input text strings.

        Returns:
            NumPy array where each row is the embedding vector for the corresponding text.
        Raises:
            APIError: If there's an issue with the OpenAI API request.
        """
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return np.array([data.embedding for data in response.data])
        except APIError as e:
            print(f"[OpenAIEmbeddingModel] API Error during batch embedding generation: {e}") # Log error with model context
            raise # Re-raise the exception


    def calculate_similarities(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate cosine similarities (using sklearn) between query and candidate embeddings.

        Args:
            query_embedding: Embedding vector for the query (NumPy array).
            candidate_embeddings: List of embedding vectors for candidates (List of NumPy arrays).

        Returns:
            NumPy array of cosine similarity scores.
        """
        return cosine_similarity([query_embedding], candidate_embeddings)[0]