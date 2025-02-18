# embedding_factory.py
from typing import Optional, Dict, Any
from .abstract_embedding import EmbeddingModel, EmbeddingError
from .sentence_transformers_embedding import SentenceTransformersEmbeddingModel
from .openai_embedding import OpenAIEmbeddingModel



class EmbeddingFactory:
    """Factory for creating embedding model instances."""

    @staticmethod
    def create_embedding_model(
        model_type: str, embedding_config: Optional[Dict[str, Any]] = None
    ) -> EmbeddingModel:
        """Create an embedding model instance of the specified type.

        Args:
            model_type: Type of embedding model to create ('sentence-transformers' or 'openai').
            embedding_config: Optional dictionary containing configuration parameters for the embedding model.

        Returns:
            Instance of the specified embedding model.

        Raises:
            ValueError: If the specified model type is not supported.
            TypeError: If embedding_config is not a dictionary when required, or if API key is missing for OpenAI.
        """
        if embedding_config is None:
            embedding_config = {}  # Use empty dict if no config provided

        model_type = model_type.lower() # Normalize model type string

        if model_type == "sentence_transformers":
            model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
            return SentenceTransformersEmbeddingModel(model_name=model_name)

        elif model_type == "openai":
            api_key = embedding_config.get("api_key")
            if not api_key:
                raise ValueError("API key is required in embedding_config for 'openai' model_type.")
            model_name = embedding_config.get("model", "text-embedding-3-small") # Consistent naming: model_name
            return OpenAIEmbeddingModel(api_key=api_key, model=model_name) # Pass model_name

        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")