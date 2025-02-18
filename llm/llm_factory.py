# llm_factory.py
from typing import Optional, Dict, Any
from .abstract_llm import LLMInterface, LLMError
from .huggingface_llm import HuggingFaceLLM
from .openai_llm import OpenAILLM


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_llm(llm_type: str, llm_config: Optional[Dict[str, Any]] = None) -> LLMInterface:
        """Create an LLM instance of the specified type.

        Args:
            llm_type: Type of LLM to create ('huggingface' or 'openai').
            llm_config: Optional dictionary containing configuration parameters for the LLM.

        Returns:
            Instance of the specified LLM.

        Raises:
            ValueError: If the specified LLM type is not supported, or if required configuration
                        parameters are missing or invalid.
            TypeError: If llm_config is not a dictionary when required.
        """
        if llm_config is None:
            llm_config = {}  # Use empty dict if no config provided

        llm_type = llm_type.lower() # Normalize LLM type string

        if llm_type == "huggingface":
            required_args = {"api_key", "model_endpoint"}
            missing_args = required_args.difference(llm_config.keys())
            if missing_args:
                raise ValueError(
                    f"Missing required configuration parameters for HuggingFace LLM in llm_config: {missing_args}. "
                    f"Required parameters are: {required_args}"
                )
            return HuggingFaceLLM(
                api_key=llm_config["api_key"],
                model_endpoint=llm_config["model_endpoint"],
                timeout=llm_config.get("timeout", 30),  # Default timeout
                max_length=llm_config.get("max_length", 2048), # Default max_length
            )

        elif llm_type == "openai":
            if "api_key" not in llm_config:
                raise ValueError("API key is required in llm_config for 'openai' LLM type.")
            return OpenAILLM(
                api_key=llm_config["api_key"],
                model=llm_config.get("model", "gpt-3.5-turbo"), # Default model
                timeout=llm_config.get("timeout", 30), # Default timeout
            )

        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")