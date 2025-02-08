# abstract_llm.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import logging

# Set up basic logging for the module (can be configured further in implementations)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    def query(
        self, prompt: str, max_retries: int = 3, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """Query the LLM with a prompt and handle retries.

        Args:
            prompt: Input text prompt to send to the LLM.
            max_retries: Maximum number of retry attempts in case of transient errors (default: 3).
            **kwargs: Additional model-specific parameters that will be passed to the underlying LLM API.
                     These might include parameters like 'temperature', 'top_p', 'max_tokens', etc.

        Returns:
            Tuple[str, Dict[str, Any]]:
                - str: The generated response text from the LLM.
                - Dict[str, Any]: Metadata dictionary containing information about the LLM query,
                  such as token usage, model information, and potentially API request details.
                  The content of this dictionary will vary depending on the specific LLM implementation.

        Raises:
            LLMError: If the query fails after all retry attempts due to a non-transient error,
                      or if the LLM service is consistently unavailable.
                      Specific subclasses of LLMError (e.g., LLMConnectionError, LLMTimeoutError)
                      may be raised to indicate the type of error more precisely.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is currently available and responding.

        This method should perform a quick check to determine if the LLM service is reachable
        and operational. Implementations might involve sending a lightweight ping request or
        checking the service status.

        Returns:
            bool: True if the LLM service is available, False otherwise.
        """
        pass


class LLMError(Exception):
    """Base exception class for LLM-related errors within the CAG framework."""

    pass


class LLMConnectionError(LLMError):
    """Raised when a connection to the LLM service cannot be established."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when a request to the LLM service times out."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM service returns a rate limit error."""

    pass