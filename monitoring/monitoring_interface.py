# monitoring/monitoring_interface.py
from abc import ABC, abstractmethod
from typing import Optional, Dict

class MonitoringInterface(ABC):
    """Abstract base class for defining monitoring interfaces for the CAG framework.

    This interface outlines the methods that concrete monitoring implementations should provide
    to track and report on the performance and operational metrics of the Cache-Augmented
    Generation (CAG) engine. Implementations can range from simple console logging to
    more sophisticated systems that export metrics to monitoring dashboards or databases.
    """

    @abstractmethod
    def increment_cache_hits(self) -> None:
        """Increment the counter for cache hits.

        This method should be called whenever a query is successfully served from the cache
        (exact match). Concrete implementations should handle the actual incrementing of the counter.
        """
        pass

    @abstractmethod
    def increment_cache_misses(self) -> None:
        """Increment the counter for cache misses.

        This method should be called whenever a query is not found in the cache and requires
        processing by the LLM. Concrete implementations should handle the counter increment.
        """
        pass

    @abstractmethod
    def increment_semantic_hits(self) -> None:
        """Increment the counter for semantic cache hits.

        This method should be called when a query is served from the cache based on semantic similarity
        to a previously cached query. Concrete implementations are responsible for incrementing the counter.
        """
        pass

    @abstractmethod
    def record_response_time(self, time_seconds: float) -> None:
        """Record the response time for a query.

        This method should be called after processing a query (whether from cache or LLM) to record
        the total time taken to generate a response.

        Args:
            time_seconds: The response time in seconds (float). Must be a non-negative value.

        Raises:
            ValueError: if time_seconds is negative.
        """
        if time_seconds < 0:
            raise ValueError("Response time must be a non-negative value.")
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, float]:
        """Retrieve the current monitoring statistics as a dictionary.

        This method should return a dictionary containing the current values of all tracked metrics.
        The dictionary keys should be descriptive names for the metrics (e.g., 'cache_hits', 'cache_misses',
        'avg_response_time'). The values should be the corresponding metric values (typically numbers).

        Returns:
            Dict[str, float]: A dictionary containing monitoring statistics.
                               The keys are metric names (strings), and the values are metric values (floats or ints).
                               Example keys might include: 'total_queries', 'cache_hits', 'cache_misses',
                               'semantic_hits', 'hit_rate', 'avg_response_time', 'uptime_seconds'.
        """
        pass