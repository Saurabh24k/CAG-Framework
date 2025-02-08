# openai_llm.py
from typing import Dict, Any, Tuple, Optional
import time
import logging
from openai import OpenAI
from openai import APIError, RateLimitError, OpenAIError, Timeout  # Import specific OpenAI errors

from .abstract_llm import LLMInterface, LLMError, LLMConnectionError, LLMTimeoutError, LLMRateLimitError

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """OpenAI API implementation for Chat Models (like gpt-3.5-turbo, gpt-4)."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", timeout: int = 30):
        """Initialize the OpenAI LLM client.

        Args:
            api_key: OpenAI API key. Required.
            model: OpenAI Chat model identifier (e.g., "gpt-3.5-turbo", "gpt-4").
                   Defaults to "gpt-3.5-turbo".
            timeout: Request timeout in seconds for API calls (default: 30 seconds).

        Raises:
            ValueError: if api_key is not provided.
        """
        if not api_key:
            raise ValueError("OpenAI API key must be provided for OpenAILLM.")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self._model_name = model # Store model name for metadata


    def query(
        self, prompt: str, max_retries: int = 3, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """Query the OpenAI Chat API with retry logic and error handling.

        Args:
            prompt: Input text prompt for the LLM.
            max_retries: Maximum number of retry attempts for transient errors (default: 3).
            **kwargs: Additional parameters to pass to the OpenAI Chat Completions API.
                      See OpenAI API documentation for available parameters
                      (e.g., 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', etc.).

        Returns:
            Tuple[str, Dict[str, Any]]:
                - str: Generated text response from the OpenAI Chat API.
                - Dict[str, Any]: Metadata dictionary containing information about the API request,
                  including provider, model name, token usage, creation timestamp, and raw response details.

        Raises:
            LLMError: If the query fails after all retry attempts due to a non-retriable error
                      or repeated transient errors exceeding max_retries.
        """
        last_error = None
        metadata = {"provider": "openai", "model": self._model_name} # Use stored model name

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs
                )

                metadata.update(
                    {
                        "usage": response.usage.model_dump(), # Capture token usage
                        "created": response.created, # Response creation timestamp
                        "raw_response": response.model_dump() # Include raw response for detailed debugging
                    }
                )
                return response.choices[0].message.content, metadata # Extract response content and return

            except RateLimitError as e: # OpenAI RateLimitError
                logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}, waiting before retry: {e}")
                time.sleep(2**attempt) # Exponential backoff
                last_error = e
            except Timeout as e: # OpenAI Timeout
                logger.warning(f"Timeout error on attempt {attempt + 1}/{max_retries}: {str(e)}, retrying...")
                time.sleep(1)
                last_error = e
            except APIError as e: # General OpenAI API errors (e.g., authentication, invalid requests)
                logger.error(f"OpenAI API error (non-retriable): {str(e)}")
                raise # Re-raise - these are generally not transient
            except OpenAIError as e: # Catch-all for other OpenAI related errors
                logger.error(f"OpenAI SDK error (non-retriable): {str(e)}")
                raise # Re-raise - assume non-recoverable for now
            except Exception as e: # Unexpected errors
                logger.error(f"Unexpected error during OpenAI query: {str(e)}")
                raise LLMError(f"Unexpected error during OpenAI query: {str(e)}") from e # Wrap in LLMError

        # If loop completes without success, raise a final LLMError
        raise LLMError(f"OpenAI query failed after {max_retries} attempts. Last error: {str(last_error)}")



    def is_available(self) -> bool:
        """Check if the OpenAI API is available by making a lightweight test query."""
        try:
            # Use a very simple, cheap query to check availability
            self.query("Is the API available?", max_retries=1, max_tokens=5) # Minimal query
            return True # If query succeeds, API is likely available
        except Exception: # Catch any exception during the test query
            return False # API is considered unavailable if any error occurs