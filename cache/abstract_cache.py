# abstract_cache.py
from abc import ABC, abstractmethod
from typing import Any, Optional, List
import numpy as np

class CacheError(Exception):
    """Base exception for cache-related errors."""
    pass

class AbstractCache(ABC):
    """Abstract base class defining the interface for cache implementations."""

    @abstractmethod
    def add(self, key: str, value: Any, embedding: Optional[np.ndarray] = None) -> None:
        """Add an item to the cache.

        Args:
            key: The key to store the value under.
            value: The value to cache.
            embedding: Optional embedding vector (numpy array) associated with the value.
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from the cache.

        Args:
            key: The key to look up.

        Returns:
            The cached value if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retrieve the embedding for a cached item.

        Args:
            key: The key to look up.

        Returns:
            The embedding (numpy array) if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_embeddings_batch(self, keys: List[str]) -> List[np.ndarray]:
        """Retrieve a batch of embeddings for given keys.

        Args:
            keys: List of keys to look up.

        Returns:
            List of embeddings (numpy arrays). Returns an empty list if no embeddings are found or if keys are invalid.
            Implementations should handle cases where some keys have embeddings and others do not, returning None for missing embeddings in the list, or filtering out keys without embeddings.
        """
        pass


    @abstractmethod
    def get_keys(self) -> List[str]:
        """Retrieve all keys currently in the cache.

        Returns:
            List of keys (strings).
        """
        pass


    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the cache."""
        pass

    @abstractmethod
    def evict(self) -> None:
        """Evict item(s) from the cache according to the implementation's policy."""
        pass

