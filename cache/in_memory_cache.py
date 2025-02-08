# in_memory_cache.py
import time
import threading
from typing import Any, Dict, Optional

from .abstract_cache import AbstractCache


class InMemoryCache(AbstractCache):
    """Thread-safe in-memory cache implementation with LRU eviction."""

    def __init__(self, max_size: int = 100):
        """Initialize the cache.

        Args:
            max_size: Maximum number of items to store in the cache.
        """
        self._cache: Dict[str, Dict] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

    def _normalize_key(self, key: str) -> str:
        """Normalize the cache key (strip and lowercase).

        Args:
            key: The key to normalize.

        Returns:
            Normalized key string.
        """
        return key.strip().lower()

    def add(self, key: str, value: Any, embedding: Optional[Any] = None) -> None:
        """Add an item to the cache. Evicts LRU item if cache is full."""
        normalized_key = self._normalize_key(key)
        with self._lock:
            if len(self._cache) >= self._max_size:
                self.evict()
            self._cache[normalized_key] = {
                "response": value,
                "timestamp": time.time(),
                "embedding": embedding,
            }

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from the cache."""
        normalized_key = self._normalize_key(key)
        with self._lock:
            cache_entry = self._cache.get(normalized_key)
            if cache_entry:
                # Update timestamp on cache hit for LRU
                cache_entry["timestamp"] = time.time()
                return cache_entry["response"]
            return None

    def get_embedding(self, key: str) -> Optional[Any]:
        """Retrieve the embedding for a cached item."""
        normalized_key = self._normalize_key(key)
        with self._lock:
            return self._cache.get(normalized_key, {}).get("embedding", None)

    def clear(self) -> None:
        """Remove all items from the cache."""
        with self._lock:
            self._cache.clear()

    def evict(self) -> None:
        """Evict the least recently used item from the cache (LRU)."""
        with self._lock:
            if self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
                del self._cache[oldest_key]