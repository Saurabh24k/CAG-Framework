# abstract_embedding.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Union, Tuple

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for input text.

        Args:
            text: Input text to generate embedding for.

        Returns:
            Numpy array containing the embedding vector.
        """
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts.

        Returns:
            Numpy array where each row is an embedding vector for the corresponding input text.
        """
        pass

    @abstractmethod
    def calculate_similarities(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate similarity scores between a query embedding and a list of candidate embeddings.

        Args:
            query_embedding: Embedding vector for the query.
            candidate_embeddings: List of embedding vectors for candidates.

        Returns:
            Numpy array of similarity scores (e.g., cosine similarity) between the query embedding
            and each candidate embedding.
        """
        pass


    def find_best_match(
        self,
        query: Union[str, np.ndarray],
        candidates: List[Union[str, np.ndarray]],
        threshold: float = 0.8,
    ) -> Tuple[int, float]:
        """Find the best matching candidate for a query based on embedding similarity.

        This method can accept either the query text or its pre-computed embedding, and a list of
        candidate texts or their pre-computed embeddings. It automatically generates embeddings
        if text inputs are provided. It then calculates similarities and returns the index and score
        of the best match that exceeds the given threshold.

        Args:
            query: Query text (str) or its embedding (np.ndarray).
            candidates: List of candidate texts (str) or their embeddings (np.ndarray).
            threshold: Minimum similarity score to consider a match (default: 0.8).

        Returns:
            Tuple[int, float]: (best match index, similarity score).
            Returns (-1, 0.0) if no match is found above the threshold or if there are no candidates.
        """
        # Convert query to embedding if text is provided
        if isinstance(query, str):
            query_embedding = self.generate_embedding(query)
        elif isinstance(query, np.ndarray):
            query_embedding = query
        else:
            raise TypeError("Query must be either a string or a NumPy array embedding.")


        # Convert candidates to embeddings if texts are provided
        candidate_embeddings = []
        for candidate in candidates:
            if isinstance(candidate, str):
                candidate_embeddings.append(self.generate_embedding(candidate))
            elif isinstance(candidate, np.ndarray):
                candidate_embeddings.append(candidate)
            else:
                raise TypeError("Candidates must be strings or NumPy array embeddings.")


        if not candidate_embeddings:
            return -1, 0.0  # No candidates to compare against

        similarities = self.calculate_similarities(query_embedding, candidate_embeddings)
        best_match_index = int(np.argmax(similarities)) # Ensure index is int
        best_match_score = float(similarities[best_match_index]) # Ensure score is float

        if best_match_score >= threshold:
            return best_match_index, best_match_score
        return -1, 0.0