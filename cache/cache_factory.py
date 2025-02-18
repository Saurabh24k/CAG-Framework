# cache_factory.py
from typing import Optional, Dict, Any
from .abstract_cache import AbstractCache, CacheError
from .in_memory_cache import InMemoryCache
from .redis_cache import RedisCache



class CacheFactory:
    """Factory for creating cache instances."""

    @staticmethod
    def create_cache(cache_type: str, cache_config: Optional[Dict[str, Any]] = None) -> AbstractCache:
        """Create a cache instance of the specified type.

        Args:
            cache_type: Type of cache to create ('in-memory' or 'redis').
            cache_config: Optional dictionary containing configuration parameters for the cache.

        Returns:
            An instance of the specified cache type.

        Raises:
            ValueError: If the specified cache type is not supported.
            TypeError: If cache_config is not a dictionary when required.
        """
        if cache_config is None:
            cache_config = {}  # Use empty dict if no config provided

        cache_type = cache_type.lower() # Normalize cache type string

        if cache_type == "in_memory":
            max_size = cache_config.get("max_size", 100)  # Default max_size if not in config
            if not isinstance(max_size, int):
                raise TypeError(f"Invalid 'max_size' in cache config. Must be an integer, but got: {type(max_size)}")
            return InMemoryCache(max_size=max_size)

        elif cache_type == "redis":
            host = cache_config.get("host", "localhost")
            port = cache_config.get("port", 6379)
            db = cache_config.get("db", 0)

            if not isinstance(port, int):
                raise TypeError(f"Invalid 'port' in cache config. Must be an integer, but got: {type(port)}")
            if not isinstance(db, int):
                raise TypeError(f"Invalid 'db' in cache config. Must be an integer, but got: {type(db)}")

            return RedisCache(host=host, port=port, db=db)

        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")