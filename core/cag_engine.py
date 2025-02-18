# cag_engine.py
import logging
import time
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from functools import lru_cache

from config.config_manager import ConfigManager, ConfigurationError
from cache.cache_factory import CacheFactory, CacheError
from embedding.embedding_factory import EmbeddingFactory, EmbeddingError
from llm.llm_factory import LLMFactory, LLMError
from monitoring.monitoring_interface import MonitoringInterface
from monitoring.console_monitoring import ConsoleMonitoring # Import ConsoleMonitoring
from similarity.similarity_metric import (
    SimilarityMetric,
    CosineSimilarityMetric,
    EuclideanSimilarityMetric,
)

logger = logging.getLogger(__name__)


class CAGEngineError(Exception):
    """Base exception class for errors specifically within the CAGEngine."""

    pass


class CAGEngine:
    """Core engine for the Cache-Augmented Generation (CAG) framework.

    Orchestrates the process of handling user queries by first checking the cache for
    exact or semantically similar queries and their responses. If no suitable cache entry
    is found, it queries the configured Large Language Model (LLM), caches the LLM's response,
    and then returns the response. This engine leverages configurable components for caching,
    embedding generation, LLM interaction, similarity measurement, and monitoring.
    """

    def __init__(self, config_path: Optional[str] = None): # Removed monitoring arg from init
        """Initialize the CAGEngine by loading configuration and initializing components.

        Args:
            config_path: Optional path to the configuration YAML file. If None, it will try to
                        find 'config.yaml' in standard locations.

        Raises:
            CAGEngineError: If any part of the initialization process fails.
        """
        try:
            logger.info("Initializing CAG Engine...")

            # 1. Load Configuration
            self.config = ConfigManager(config_path)
            logger.debug("Configuration loaded successfully.")

            # 2. Initialize Monitoring - NEW SECTION
            monitoring_config = self.config.get_monitoring_config() # Get monitoring config
            monitoring_enabled = monitoring_config.get('enabled', False) # Check if monitoring is enabled (default: False)
            self.monitoring: Optional[MonitoringInterface] = None # Initialize monitoring attribute to None
            if monitoring_enabled: # Only initialize monitoring if enabled in config
                monitoring_type = monitoring_config.get('type', 'console').lower() # Get monitoring type, default to console
                if monitoring_type == 'console':
                    self.monitoring = ConsoleMonitoring() # Initialize ConsoleMonitoring
                    logger.info("Console monitoring enabled.")
                elif monitoring_type == 'custom': # Placeholder for future custom monitoring
                    logger.warning("Custom monitoring type is not yet implemented. Falling back to no monitoring.")
                    self.monitoring = None # No monitoring for now, but you could add custom factory/class here later
                else:
                    logger.warning(f"Unsupported monitoring type: {monitoring_type}. Console monitoring enabled as fallback.")
                    self.monitoring = ConsoleMonitoring() # Fallback to ConsoleMonitoring
            else:
                logger.info("Monitoring disabled as per configuration.") # Log if monitoring is disabled


            # 3. Initialize Cache
            cache_config = self.config.get_cache_config()
            self.cache = CacheFactory.create_cache(
                cache_config["type"], cache_config
            )
            logger.debug(f"Cache initialized: {type(self.cache).__name__}")

            # 4. Initialize Embedding Model
            embedding_config = self.config.get_embedding_config()
            self.embedding_model = EmbeddingFactory.create_embedding_model(
                embedding_config["type"], embedding_config
            )
            logger.debug(f"Embedding model initialized: {type(self.embedding_model).__name__}")

            # 5. Initialize LLM
            llm_config = self.config.get_llm_config()
            self.llm = LLMFactory.create_llm(
                llm_config["type"], llm_config
            )
            logger.debug(f"LLM initialized: {type(self.llm).__name__}")

            # 6. Initialize Similarity Metric
            similarity_config = self.config.get_similarity_config()
            self.similarity_metric = self._create_similarity_metric(
                similarity_config["metric"], similarity_config
            )
            self.similarity_threshold = similarity_config.get("threshold", 0.8)
            logger.debug(
                f"Similarity metric initialized: {type(self.similarity_metric).__name__}, "
                f"Threshold: {self.similarity_threshold}"
            )

            logger.info("CAG Engine initialized successfully.")

        except ConfigurationError as e:
            logger.error(f"Configuration error during CAG Engine initialization: {e}")
            raise CAGEngineError(f"Configuration error: {e}") from e
        except CacheError as e:
            logger.error(f"Cache initialization error: {e}")
            raise CAGEngineError(f"Cache initialization failed: {e}") from e
        except EmbeddingError as e:
            logger.error(f"Embedding model initialization error: {e}")
            raise CAGEngineError(f"Embedding model initialization failed: {e}") from e
        except LLMError as e:
            logger.error(f"LLM initialization error: {e}")
            raise CAGEngineError(f"LLM initialization failed: {e}") from e
        except ValueError as e:
            logger.error(f"Similarity metric initialization error: {e}")
            raise CAGEngineError(f"Similarity metric initialization failed: {e}") from e
        except Exception as e:
            logger.critical(f"Unexpected error during CAG Engine initialization: {e}", exc_info=True)
            raise CAGEngineError(f"Failed to initialize CAG Engine: {e}") from e

    def _create_similarity_metric(self, metric_type: str, config: Dict[str, Any]) -> SimilarityMetric:
        """Create and configure a SimilarityMetric instance.

        Args:
            metric_type: String identifier for the similarity metric ('cosine' or 'euclidean').
            config: Dictionary containing the similarity configuration.

        Returns:
            SimilarityMetric: Configured similarity metric instance.

        Raises:
            ValueError: If metric_type is not supported.
        """
        metric_type = metric_type.lower()
        if metric_type == "cosine":
            return CosineSimilarityMetric()
        elif metric_type == "euclidean":
            sigma = config.get("euclidean", {}).get("sigma", 1.0)
            return EuclideanSimilarityMetric(sigma=sigma)
        else:
            raise ValueError(
                f"Unsupported similarity metric type: {metric_type}. "
                "Valid types are: 'cosine', 'euclidean'."
            )

    @lru_cache(maxsize=1)
    def _normalize_query(self, query: str) -> str:
        """Normalize the input query string for consistent cache lookups.

        Args:
            query: The original user query string.

        Returns:
            str: The normalized query string.
        """
        return query.strip().lower()

    def _find_similar_query(
        self, query_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Find the most semantically similar query from the cache.

        Args:
            query_embedding: NumPy array representing the embedding vector of the current query.

        Returns:
            Tuple[Optional[str], float]: The most similar cached query and its similarity score.
        """
        cached_queries: List[str] = self.cache.get_keys() # Get all cached query keys
        if not cached_queries:
            return None, 0.0

        cached_embeddings: List[np.ndarray] = self.cache.get_embeddings_batch(cached_queries) # Get embeddings for all cached queries
        if not cached_embeddings: # Handle case where no embeddings are in cache, but keys exist (though unlikely if cache is used correctly)
            return None, 0.0

        similarities: np.ndarray = self.similarity_metric.calculate_similarities(
            query_embedding, cached_embeddings
        )

        best_match_index: int = int(np.argmax(similarities))
        best_match_score: float = float(similarities[best_match_index])

        if best_match_score >= self.similarity_threshold:
            best_match_query: str = cached_queries[best_match_index]
            logger.debug(
                f"Semantic cache match found. Score: {best_match_score:.3f}, "
                f"Query: '{best_match_query[:50]}...'"
            )
            return best_match_query, best_match_score
        return None, 0.0


    def generate_response(self, normalized_query: str) -> Tuple[str, str]:
        """Process a normalized user query using the Cache-Augmented Generation pipeline and return raw response without prefix.

        Args:
            normalized_query: The normalized user query string.

        Returns:
            Tuple[str, str]: Tuple containing the response string and the cache status ('Hit' or 'Miss').

        Raises:
            CAGEngineError: If any error occurs during query processing.
        """
        start_time = time.time()
        # 1. Check for Exact Cache Hit
        cached_response: Optional[str] = self.cache.get(normalized_query)
        if cached_response is not None:
            logger.info("Exact cache hit for query: '%s'", normalized_query)
            if self.monitoring:  # Check if monitoring is enabled
                self.monitoring.increment_cache_hits()  # Increment cache hits
                self.monitoring.record_response_time(time.time() - start_time)  # Record response time (cache lookup time is negligible)
            return cached_response, "ExactHit"

        # 2. Generate Embedding for Query
        query_embedding: np.ndarray = self.embedding_model.generate_embedding(normalized_query)
        logger.debug(
            f"Generated embedding for query: '{normalized_query[:50]}...', "
            f"shape: {query_embedding.shape}"
        )

        # 3. Check for Semantic Cache Match
        similar_query, similarity_score = self._find_similar_query(query_embedding)

        if similar_query is not None:
            cached_response = self.cache.get(similar_query)
            logger.info(
                "Semantic cache hit for query: '%s' (matched: '%s', score: %.3f)",
                normalized_query, similar_query, similarity_score
            )
            if self.monitoring:  # Check if monitoring is enabled
                self.monitoring.increment_semantic_hits()  # Increment semantic hits
                self.monitoring.record_response_time(time.time() - start_time)  # Record response time
            return cached_response, "SemanticHit"

        # 4. Query LLM for Response (Cache Miss)
        logger.info("Cache miss (exact & semantic), querying LLM for: '%s'", normalized_query)
        llm_response: str = ""
        metadata: Dict[str, Any] = {}
        llm_start_time = time.time()  # Start time for LLM query
        llm_response, metadata = self.llm.query(normalized_query)
        llm_end_time = time.time()  # End time for LLM query
        llm_response_time = llm_end_time - llm_start_time  # LLM query time
        logger.debug(
            f"LLM response received for query: '{normalized_query[:50]}...', "
            f"response[:50]: '{llm_response[:50]}...', LLM response time: {llm_response_time:.3f}s"
        )

        # 5. Cache the LLM Response with Embedding
        self.cache.add(
            normalized_query, llm_response, embedding=query_embedding
        )  # Store embedding along with response
        logger.debug(f"Response cached for query: '{normalized_query[:50]}...'")

        if self.monitoring:  # Check if monitoring is enabled
            self.monitoring.increment_cache_misses()  # Increment cache misses
            self.monitoring.record_response_time(time.time() - start_time)  # Record total response time

        # 6. Return LLM Response (Cache Miss)
        return llm_response, "Miss" # Indicate cache miss



    def query(self, user_query: str) -> str:
        """Process a user query using the Cache-Augmented Generation pipeline.

        Args:
            user_query: The original user query string.

        Returns:
            str: The response string, prefixed with cache status.

        Raises:
            CAGEngineError: If any error occurs during query processing.
        """
        try:

            # 1. Normalize Query
            normalized_query: str = self._normalize_query(user_query)
            logger.debug(f"Normalized query: '{normalized_query}'")

            llm_response, cache_status = self.generate_response(normalized_query)

            if cache_status == "ExactHit":
                return f"Cache Hit! {llm_response}"
            elif cache_status == "SemanticHit":
                return f"Semantic Cache Hit! {llm_response}"
            elif cache_status == "Miss":
                return f"Cache Miss! {llm_response}"
            else: # Should not reach here, but for safety
                return llm_response


        except CAGEngineError as e:
            raise e # Re-raise CAGEngineError directly
        except CacheError as e:
            logger.error(f"Cache error during query processing: {e}")
            raise CAGEngineError(f"Cache operation failed: {e}") from e
        except EmbeddingError as e:
            logger.error(f"Embedding error during query processing: {e}")
            raise CAGEngineError(f"Embedding generation failed: {e}") from e
        except LLMError as e:
            logger.error(f"LLM query error during query processing: {e}")
            raise CAGEngineError(f"LLM query failed: {e}") from e
        except Exception as e:
            logger.critical(f"Unexpected error during query processing: {e}", exc_info=True)
            raise CAGEngineError(f"Unexpected error during query processing: {e}") from e


    def refresh_cache(self) -> None:
        """Clear the entire cache.

        Raises:
            CAGEngineError: If there's an error during the cache clearing process.
        """
        try:
            self.cache.clear()
            logger.info("Cache cleared successfully.")
        except CacheError as e:
            logger.error(f"Failed to clear cache: {e}")
            raise CAGEngineError(f"Failed to clear cache: {e}") from e
        except Exception as e:
            logger.critical(f"Unexpected error during cache clearing: {e}", exc_info=True)
            raise CAGEngineError(f"Unexpected error during cache refresh: {e}") from e

