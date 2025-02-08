# similarity_metric.py
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class SimilarityMetric(ABC):
    """Abstract base class for similarity metrics used in Cache-Augmented Generation."""

    @abstractmethod
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate the similarity score between two embedding vectors.

        This method should be implemented by concrete subclasses to define specific
        similarity measures (e.g., cosine similarity, Euclidean distance similarity).

        Args:
            embedding1: The first embedding vector (NumPy array). Must be 1-dimensional.
            embedding2: The second embedding vector (NumPy array). Must be 1-dimensional.

        Returns:
            float: Similarity score between the two embeddings, typically in the range [0, 1]
                   where higher values indicate greater similarity.

        Raises:
            ValueError: If embeddings have incompatible shapes (e.g., not 1-dimensional).
        """
        pass

    @abstractmethod
    def calculate_similarities(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate similarity scores between a query embedding and a list of candidate embeddings.

        This method calculates the similarity between a single query embedding and multiple
        candidate embeddings. It should return an array of similarity scores, one for each candidate.

        Args:
            query_embedding: The embedding vector for the query (NumPy array). Must be 1-dimensional.
            candidate_embeddings: A list of embedding vectors for candidates (List of NumPy arrays).
                                  Each embedding in the list must be 1-dimensional.

        Returns:
            np.ndarray: A NumPy array of similarity scores. Each element in the array represents
                          the similarity score between the query embedding and the corresponding
                          candidate embedding in the input list.

        Raises:
            ValueError: If embeddings have incompatible shapes (e.g., query not 1D or candidates not a list of 1D).
        """
        pass

    def find_best_match(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray], threshold: float = 0.8
    ) -> Tuple[int, float]:
        """Find the best matching candidate from a list based on similarity to a query embedding.

        This method calculates similarity scores between the query embedding and all candidate embeddings,
        identifies the candidate with the highest similarity score, and checks if this score exceeds
        the given threshold.

        Args:
            query_embedding: The embedding vector for the query (NumPy array).
            candidate_embeddings: A list of embedding vectors for candidates (List of NumPy arrays).
            threshold: The minimum similarity score required to consider a candidate a "match" (default: 0.8).

        Returns:
            Tuple[int, float]:
                - int: The index of the best matching candidate in the `candidate_embeddings` list.
                       Returns -1 if no candidate's similarity score is above the threshold or if
                       the `candidate_embeddings` list is empty.
                - float: The similarity score of the best match. Returns 0.0 if no match is found above the threshold.
        """
        if not candidate_embeddings:
            return -1, 0.0  # No candidates to compare against

        similarities = self.calculate_similarities(query_embedding, candidate_embeddings)
        best_match_index = int(np.argmax(similarities)) # Ensure index is int
        best_match_score = float(similarities[best_match_index]) # Ensure score is float

        if best_match_score >= threshold:
            return best_match_index, best_match_score
        return -1, 0.0



class CosineSimilarityMetric(SimilarityMetric):
    """Cosine Similarity Metric implementation, inheriting from SimilarityMetric."""

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embedding vectors.

        Cosine similarity measures the cosine of the angle between two vectors.
        It ranges from -1 to 1, but for embeddings representing text, scores typically fall
        in the range [0, 1], with 1 indicating identical embeddings and 0 indicating no similarity.

        Args:
            embedding1: The first embedding vector (NumPy array). Must be 1-dimensional.
            embedding2: The second embedding vector (NumPy array). Must be 1-dimensional.

        Returns:
            float: Cosine similarity score between embedding1 and embedding2 (range [0, 1]).

        Raises:
            ValueError: If input embeddings are not 1-dimensional NumPy arrays or have incompatible shapes.
        """
        # Input validation: Ensure embeddings are 1D NumPy arrays
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise ValueError("Input embeddings must be NumPy arrays.")
        if embedding1.ndim != 1 or embedding2.ndim != 1:
            raise ValueError("Embeddings must be 1-dimensional arrays.")
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding shapes are incompatible. Got {embedding1.shape} and {embedding2.shape}")


        # Reshape 1D embeddings to 2D arrays as required by sklearn's cosine_similarity
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)

        return float(cosine_similarity(e1, e2)[0, 0]) # Extract scalar similarity value


    def calculate_similarities(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate cosine similarity scores between a query embedding and a list of candidate embeddings.

        Args:
            query_embedding: The embedding vector for the query (NumPy array). Must be 1-dimensional.
            candidate_embeddings: A list of embedding vectors for candidates (List of NumPy arrays).
                                  Each embedding in the list must be 1-dimensional.

        Returns:
            np.ndarray: A NumPy array of cosine similarity scores.

        Raises:
            ValueError: If the query embedding is not 1-dimensional or if candidate embeddings
                        are not a list of 1-dimensional NumPy arrays.
        """
        # Input validation: Ensure correct embedding dimensions and types
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("Query embedding must be a NumPy array.")
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be a 1-dimensional array.")
        if not isinstance(candidate_embeddings, list) or not all(isinstance(emb, np.ndarray) for emb in candidate_embeddings):
            raise ValueError("Candidate embeddings must be a list of NumPy arrays.")
        for emb in candidate_embeddings:
            if emb.ndim != 1:
                raise ValueError("Each candidate embedding must be a 1-dimensional array.")
            if emb.shape != query_embedding.shape: # Shape consistency check
                raise ValueError(f"Candidate embedding shape {emb.shape} is incompatible with query embedding shape {query_embedding.shape}.")


        # Reshape query embedding to 2D array for cosine_similarity
        query_reshaped = query_embedding.reshape(1, -1)
        # Convert list of candidate embeddings to a 2D NumPy array
        candidates_array = np.array(candidate_embeddings)

        return cosine_similarity(query_reshaped, candidates_array)[0] # Extract 1D similarity scores array



class EuclideanSimilarityMetric(SimilarityMetric):
    """Euclidean Distance Similarity Metric, using a Gaussian kernel to convert distance to similarity.

    Inherits from SimilarityMetric. Converts Euclidean distance to a similarity score in the range [0, 1]
    using a Gaussian (Radial Basis Function) kernel. Similarity decreases as Euclidean distance increases.
    """

    def __init__(self, sigma: float = 1.0):
        """Initialize EuclideanSimilarityMetric with the Gaussian kernel width (sigma).

        Args:
            sigma: The standard deviation (width) of the Gaussian kernel.
                   A higher sigma value results in a slower decay of similarity with increasing distance,
                   making the similarity metric more forgiving of larger distances.
                   Defaults to 1.0. Must be a positive float.

        Raises:
            ValueError: if sigma is not a positive float.
        """
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("Sigma must be a positive float or integer.")
        self.sigma = float(sigma) # Ensure sigma is float


    def _distance_to_similarity(self, distance: float) -> float:
        """Convert Euclidean distance to a similarity score using a Gaussian kernel (RBF).

        Similarity = exp(-distance^2 / (2 * sigma^2))
        This converts distance to a similarity score between 0 and 1.

        Args:
            distance: Euclidean distance value (float, non-negative).

        Returns:
            float: Similarity score, ranging from 1 (distance=0) to approaching 0 as distance increases.
        """
        return float(np.exp(-(distance**2) / (2 * self.sigma**2)))


    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance-based similarity between two embeddings.

        Computes the Euclidean distance and then converts it to a similarity score using
        the Gaussian kernel defined by 'sigma'.

        Args:
            embedding1: The first embedding vector (NumPy array). Must be 1-dimensional.
            embedding2: The second embedding vector (NumPy array). Must be 1-dimensional.

        Returns:
            float: Euclidean distance-based similarity score (range [0, 1]).

        Raises:
            ValueError: If input embeddings are not 1-dimensional NumPy arrays or have incompatible shapes.
        """
        # Input validation: Ensure embeddings are 1D NumPy arrays
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise ValueError("Input embeddings must be NumPy arrays.")
        if embedding1.ndim != 1 or embedding2.ndim != 1:
            raise ValueError("Embeddings must be 1-dimensional arrays.")
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding shapes are incompatible. Got {embedding1.shape} and {embedding2.shape}")


        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)

        distance = float(euclidean_distances(e1, e2)[0, 0]) # Calculate Euclidean distance
        return self._distance_to_similarity(distance) # Convert distance to similarity


    def calculate_similarities(
        self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate Euclidean distance-based similarity scores between a query and a list of candidates.

        Args:
            query_embedding: The embedding vector for the query (NumPy array). Must be 1-dimensional.
            candidate_embeddings: List of embedding vectors for candidates (List of NumPy arrays).
                                  Each embedding in the list must be 1-dimensional.

        Returns:
            np.ndarray: A NumPy array of Euclidean distance-based similarity scores.

        Raises:
            ValueError: If the query embedding is not 1-dimensional or if candidate embeddings
                        are not a list of 1-dimensional NumPy arrays.
        """
        # Input validation: Ensure correct embedding dimensions and types
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("Query embedding must be a NumPy array.")
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be a 1-dimensional array.")
        if not isinstance(candidate_embeddings, list) or not all(isinstance(emb, np.ndarray) for emb in candidate_embeddings):
            raise ValueError("Candidate embeddings must be a list of NumPy arrays.")
        for emb in candidate_embeddings:
            if emb.ndim != 1:
                raise ValueError("Each candidate embedding must be a 1-dimensional array.")
            if emb.shape != query_embedding.shape: # Shape consistency check
                raise ValueError(f"Candidate embedding shape {emb.shape} is incompatible with query embedding shape {query_embedding.shape}.")


        query_reshaped = query_embedding.reshape(1, -1)
        candidates_array = np.array(candidate_embeddings)

        distances = euclidean_distances(query_reshaped, candidates_array)[0] # Calculate Euclidean distances

        # Convert distances to similarity scores using Gaussian kernel
        return np.array([self._distance_to_similarity(d) for d in distances])