# redis_cache.py
from typing import Any, Optional

import redis

from .abstract_cache import AbstractCache


class RedisCache(AbstractCache):
    """Redis-based cache implementation."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """Initialize Redis connection.

        Args:
            host: Redis host address.
            port: Redis port number.
            db: Redis database number.
        Raises:
            redis.exceptions.ConnectionError: If connection to Redis fails.
        """
        try:
            self._redis = redis.Redis(host=host, port=port, db=db)
            self._redis.ping()  # Check connection
        except redis.exceptions.ConnectionError as e:
            raise redis.exceptions.ConnectionError(f"Failed to connect to Redis: {e}") from e
        self._embedding_prefix = "emb:"

    def _normalize_key(self, key: str) -> str:
        """Normalize the cache key (strip and lowercase)."""
        return key.strip().lower()

    def add(self, key: str, value: Any, embedding: Optional[Any] = None) -> None:
        """Add an item to the Redis cache.

        Args:
            key: The key to store the value under.
            value: The value to cache (will be serialized to string).
            embedding: Optional embedding vector associated with the value (will be serialized to string).
        """
        normalized_key = self._normalize_key(key)
        try:
            self._redis.set(normalized_key, value)  # Redis stores values as bytes/strings
            if embedding is not None:
                self._redis.set(
                    f"{self._embedding_prefix}{normalized_key}", self._serialize_embedding(embedding)
                )  # Serialize embedding
        except redis.exceptions.RedisError as e:
            print(f"[WARNING] Redis ADD error for key '{key}': {e}") # Log error, don't raise to keep cache resilient

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from the Redis cache.

        Returns: Cached value (deserialized from string) or None if not found.
        """
        normalized_key = self._normalize_key(key)
        try:
            value_bytes = self._redis.get(normalized_key)
            if value_bytes:
                return value_bytes.decode('utf-8') # Decode bytes to string
            return None
        except redis.exceptions.RedisError as e:
            print(f"[WARNING] Redis GET error for key '{key}': {e}") # Log error, return None
            return None

    def get_embedding(self, key: str) -> Optional[Any]:
        """Retrieve the embedding for a cached item from Redis.

        Returns: Embedding (deserialized) or None if not found.
        """
        normalized_key = self._normalize_key(key)
        try:
            embedding_bytes = self._redis.get(f"{self._embedding_prefix}{normalized_key}")
            if embedding_bytes:
                return self._deserialize_embedding(embedding_bytes) # Deserialize embedding
            return None
        except redis.exceptions.RedisError as e:
            print(f"[WARNING] Redis GET_EMBEDDING error for key '{key}': {e}") # Log error, return None
            return None


    def clear(self) -> None:
        """Remove all items from the Redis cache (flush the database)."""
        try:
            self._redis.flushdb()
        except redis.exceptions.RedisError as e:
            print(f"[WARNING] Redis CLEAR error: {e}") # Log error, continue operation

    def evict(self) -> None:
        """Evict item(s) from the cache according to the implementation's policy.
        
        For Redis, eviction is primarily handled by Redis's maxmemory policy, not explicitly here.
        This method is left empty as Redis manages eviction.
        """
        # Redis handles eviction based on maxmemory policy
        pass

    def _serialize_embedding(self, embedding: Any) -> str:
        """Serialize embedding to a string format for Redis storage (e.g., comma-separated)."""
        if isinstance(embedding, list) or isinstance(embedding, tuple) or isinstance(embedding, type(iter([]))): # Check for list-like, tuple-like or iterator
            return ",".join(map(str, embedding)) # Convert list/tuple/iterator to comma-separated string
        elif isinstance(embedding, bytes): # If already bytes (unlikely from sentence-transformers, but robust)
            return embedding.decode('utf-8', errors='ignore') # Decode bytes to string, ignore errors
        elif isinstance(embedding, str): # If already string (should be handled, though unexpected)
            return embedding # Return as is
        else: # Assume numpy array or similar
            try: # Try to convert numpy array to list first for serialization
                return ",".join(map(str, embedding.tolist())) # Numpy array to list, then to string
            except: # Fallback for other types, try direct conversion to string
                return str(embedding) # Last resort, convert to string directly


    def _deserialize_embedding(self, embedding_str: bytes) -> Optional[Any]:
        """Deserialize embedding from string format retrieved from Redis."""
        if not embedding_str: # Handle None or empty bytes
            return None

        embedding_str_decoded = embedding_str.decode('utf-8', errors='ignore').strip() # Decode bytes to string and strip whitespace

        if not embedding_str_decoded: # Check if empty string after decoding and stripping
            return None

        try:
            return [float(val) for val in embedding_str_decoded.split(",") if val] # Split comma-separated string, convert to floats
        except ValueError:
            print(f"[WARNING] Embedding DESERIALIZATION error for string '{embedding_str_decoded}': Could not convert to float list.")
            return None # Handle deserialization errors robustly