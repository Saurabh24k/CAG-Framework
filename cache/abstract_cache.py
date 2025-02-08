# abstract_cache.py
from abc import ABC, abstractmethod
from typing import Any, Optional

class AbstractCache(ABC):
    """Abstract base class defining the interface for cache implementations."""

    @abstractmethod
    def add(self, key: str, value: Any, embedding: Optional[Any] = None) -> None:
        """Add an item to the cache.

        Args:
            key: The key to store the value under.
            value: The value to cache.
            embedding: Optional embedding vector associated with the value.
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
    def get_embedding(self, key: str) -> Optional[Any]:
        """Retrieve the embedding for a cached item.

        Args:
            key: The key to look up.

        Returns:
            The embedding if found, None otherwise.
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