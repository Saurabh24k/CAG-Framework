# monitoring/console_monitoring.py
import time
from typing import Dict, List, Any
from .monitoring_interface import MonitoringInterface


class ConsoleMonitoring(MonitoringInterface):
    """Console-based implementation of the MonitoringInterface for CAG engine metrics.

    This class provides a simple monitoring solution that prints real-time updates and
    periodic statistics to the console. It tracks key metrics such as cache hits, misses,
    semantic hits, response times, and overall hit rate and uptime. This implementation
    is primarily intended for development, debugging, and basic operational insights.
    For production environments, more robust monitoring solutions (e.g., Prometheus, Grafana, cloud-based monitoring)
    would typically be preferred.
    """

    def __init__(self):
        """Initialize ConsoleMonitoring instance.

        Sets up counters for cache hits, misses, semantic hits, initializes an empty list
        to store response times, and records the start time for uptime calculation.
        """
        self.cache_hits: int = 0  # Counter for exact cache hits
        self.cache_misses: int = 0 # Counter for cache misses (LLM queries)
        self.semantic_hits: int = 0 # Counter for semantic cache hits
        self.response_times: List[float] = [] # List to store response times in seconds
        self.start_time: float = time.time() # Record start time for uptime calculation


    def increment_cache_hits(self) -> None:
        """Increment the cache hits counter and print a console update.

        Called when a query is served from the exact cache match. It increments the 'cache_hits' counter
        and triggers an update print to the console displaying the latest monitoring statistics.
        """
        self.cache_hits += 1 # Increment cache hit counter
        self._print_update("Cache Hit") # Trigger console update


    def increment_cache_misses(self) -> None:
        """Increment the cache misses counter and print a console update.

        Called when a query results in a cache miss and requires an LLM query.
        It increments the 'cache_misses' counter and triggers a console update.
        """
        self.cache_misses += 1 # Increment cache miss counter
        self._print_update("Cache Miss") # Trigger console update


    def increment_semantic_hits(self) -> None:
        """Increment the semantic cache hits counter and print a console update.

        Called when a query is served from the semantic cache (similar query found).
        It increments the 'semantic_hits' counter and triggers a console update.
        """
        self.semantic_hits += 1 # Increment semantic hit counter
        self._print_update("Semantic Cache Hit") # Trigger console update


    def record_response_time(self, time_seconds: float) -> None:
        """Record the response time for a query and print a console update.

        Stores the response time (in seconds) for a completed query in the 'response_times' list
        and triggers a console update.

        Args:
            time_seconds: The response time in seconds (float). Must be a non-negative value.

        Raises:
            ValueError: if time_seconds is negative (validation inherited from MonitoringInterface).
        """
        super().record_response_time(time_seconds) # Validate time_seconds using parent class validation
        self.response_times.append(time_seconds) # Append response time to the list
        self._print_update(f"Response Time: {time_seconds:.3f}s") # Trigger console update


    def get_stats(self) -> Dict[str, Any]:
        """Calculate and return the current monitoring statistics as a dictionary.

        Computes statistics such as total queries processed, hit rate (percentage of cache hits),
        average response time, and uptime.

        Returns:
            Dict[str, Any]: A dictionary containing the current monitoring statistics.
                             Keys include: 'total_queries', 'cache_hits', 'cache_misses', 'semantic_hits',
                             'hit_rate' (as a float between 0 and 1), 'avg_response_time', and 'uptime_seconds'.
        """
        total_queries: int = self.cache_hits + self.cache_misses + self.semantic_hits # Calculate total queries
        avg_response_time: float = sum(self.response_times) / len(self.response_times) if self.response_times else 0 # Calculate average response time (avoid division by zero)

        stats: Dict[str, Any] = { # Create stats dictionary
            "total_queries": total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate": (self.cache_hits + self.semantic_hits) / total_queries if total_queries > 0 else 0, # Calculate hit rate, handle zero total queries
            "avg_response_time": avg_response_time,
            "uptime_seconds": time.time() - self.start_time # Calculate uptime in seconds
        }
        return stats # Return the statistics dictionary


    def _print_update(self, message: str) -> None:
        """Print a formatted monitoring update to the console.

        Retrieves the latest statistics using 'get_stats()' and prints them to the console
        in a user-friendly, formatted way, preceded by a timestamp and an optional message.

        Args:
            message: A descriptive message to include in the console output (e.g., "Cache Hit", "Response Time").
        """
        stats: Dict[str, Any] = self.get_stats() # Get current stats

        # Formatted console output with clear labels and values
        output = f"""
[{time.strftime("%Y-%m-%d %H:%M:%S")}] Monitoring Update: {message}
------------------------------------
Total Queries:     {stats['total_queries']:<5}
Cache Hits:        {stats['cache_hits']:<5}
Semantic Hits:     {stats['semantic_hits']:<5}
Cache Misses:      {stats['cache_misses']:<5}
Hit Rate:          {stats['hit_rate']:.2%}
Avg Response Time: {stats['avg_response_time']:.3f}s
Uptime:            {stats['uptime_seconds']:.1f}s
------------------------------------
"""
        print(output) # Print the formatted output to console