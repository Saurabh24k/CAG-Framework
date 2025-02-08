# huggingface_llm.py
import time
import logging
from typing import Dict, Any, Tuple, Optional
import requests
from requests.exceptions import RequestException, Timeout

from .abstract_llm import (
    LLMInterface,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMRateLimitError,
)

logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLMInterface):
    """HuggingFace Inference API implementation for LLMs."""

    def __init__(
        self,
        api_key: str,
        model_endpoint: str,
        timeout: int = 30,
        max_length: int = 2048,
    ):
        """Initialize the HuggingFace LLM client.

        Args:
            api_key: HuggingFace API key. Required for authentication.
            model_endpoint: Full URL of the HuggingFace Inference API model endpoint.
                            Example: "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
            timeout: Request timeout in seconds for API calls (default: 30 seconds).
            max_length: Maximum length of generated text to request from the API (default: 2048 tokens).

        Raises:
            ValueError: if api_key or model_endpoint is not provided.
        """
        if not api_key:
            raise ValueError("HuggingFace API key is required for HuggingFaceLLM.")
        if not model_endpoint:
            raise ValueError("HuggingFace model_endpoint is required for HuggingFaceLLM.")

        self.api_key = api_key
        self.model_endpoint = model_endpoint
        self.timeout = timeout
        self.max_length = max_length
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model_name = model_endpoint # Store endpoint as model name for metadata


    def _make_request(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make a single POST request to the HuggingFace Inference API endpoint.

        Args:
            prompt: Input text prompt for the LLM.
            **kwargs: Additional parameters to be passed in the API request JSON payload
                      (overriding default parameters like 'max_length', 'temperature', etc.).

        Returns:
            Dict[str, Any]: Parsed JSON response from the API if the request is successful.

        Raises:
            LLMRateLimitError: If the API returns a 429 status code (rate limit exceeded).
            LLMTimeoutError: If the request times out.
            LLMConnectionError: If a connection error occurs during the request.
            LLMError: For other request exceptions or non-200 status codes.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": kwargs.get("max_length", self.max_length),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "num_return_sequences": 1,
            },
        }

        try:
            response = requests.post(
                self.model_endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 429:
                raise LLMRateLimitError("HuggingFace API rate limit exceeded")

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()

        except Timeout:
            raise LLMTimeoutError(f"Request to {self.model_endpoint} timed out after {self.timeout} seconds")
        except RequestException as e:
            if "connection" in str(e).lower():
                raise LLMConnectionError(f"Connection error to {self.model_endpoint}: {str(e)}")
            raise LLMError(f"HuggingFace API request failed for {self.model_endpoint}: {str(e)}")


    def query(
        self, prompt: str, max_retries: int = 3, **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """Query the HuggingFace Inference API with retry logic for robust communication.

        Implements exponential backoff for rate limits and retries for connection/timeout errors.

        Args:
            prompt: Input text prompt for the LLM.
            max_retries: Maximum number of retry attempts in case of transient errors (default: 3).
            **kwargs: Additional parameters to pass to the _make_request method and ultimately to the API.
                      Allows for overriding default generation parameters like 'temperature', 'max_length', etc.

        Returns:
            Tuple[str, Dict[str, Any]]:
                - str: Generated text response from the HuggingFace API.
                - Dict[str, Any]: Metadata dictionary containing details about the API request,
                  including provider, model endpoint, number of attempts, and the raw API response.

        Raises:
            LLMError: If the query fails after all retry attempts due to a non-retriable error
                      or repeated transient errors exceeding max_retries.
        """
        last_error = None
        metadata = {"provider": "huggingface", "model_endpoint": self._model_name} # Use stored model name

        for attempt in range(max_retries):
            try:
                response_json = self._make_request(prompt, **kwargs) # Get JSON response
                generated_text = response_json[0]["generated_text"] # Extract generated text (assuming list response)

                metadata.update(
                    {
                        "attempts": attempt + 1,
                        "raw_response": response_json,
                    }
                )

                return generated_text, metadata  # Return on successful query

            except LLMRateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}, waiting before retry: {e}")
                time.sleep(2**attempt)  # Exponential backoff: 2, 4, 8... seconds
                last_error = e # Store last error to raise if all retries fail
            except (LLMConnectionError, LLMTimeoutError) as e:
                logger.warning(f"Transient error on attempt {attempt + 1}/{max_retries}: {str(e)}, retrying...")
                time.sleep(1) # Wait 1 second for connection/timeout issues
                last_error = e
            except LLMError as e: # Non-retriable LLMErrors (not connection, timeout, rate limit)
                logger.error(f"Non-retriable LLM error: {str(e)}")
                raise # Re-raise to stop retries immediately

        # If loop finishes without successful query, raise the last error encountered
        raise LLMError(f"HuggingFace query failed after {max_retries} attempts. Last error: {str(last_error)}")


    def is_available(self) -> bool:
        """Check the availability of the HuggingFace Inference API endpoint.

        Sends a test query to the API endpoint to check for connectivity and a valid response.

        Returns:
            bool: True if the API endpoint is available and responds successfully to a test query, False otherwise.
        """
        try:
            self._make_request("Is the API available?", max_length=10) # Lightweight test query
            return True  # If _make_request succeeds without raising exceptions, API is likely available
        except Exception: # Catch any exception during the test query (connection, timeout, API errors)
            return False # API is considered unavailable if any error occurs