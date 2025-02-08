# config_manager.py
import os
import yaml
from typing import Any, Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Base exception class for configuration-related errors in the CAG framework."""
    pass


class ConfigManager:
    """Singleton Configuration Manager for the Cache-Augmented Generation (CAG) framework.

    This class is responsible for loading, validating, and providing access to the
    CAG framework's configuration settings from a YAML file. It implements the Singleton
    design pattern to ensure that only one instance of the configuration manager exists
    throughout the application.
    """

    _instance = None  # Singleton instance holder
    _config: Dict[str, Any] = {} # Configuration data dictionary
    _config_path: Optional[str] = None # Path to the configuration file

    def __new__(cls, config_path: Optional[str] = None):
        """Create or retrieve the singleton instance of ConfigManager.

        If an instance already exists, it is returned; otherwise, a new instance is created,
        initialized, and then returned.

        Args:
            config_path: Optional path to the configuration file. If not provided, the manager
                         will attempt to find the configuration file in default locations.

        Returns:
            ConfigManager: The singleton instance of the ConfigManager.
        """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls) # Create new instance only if none exists
            cls._instance._initialize(config_path) # Initialize on creation
        return cls._instance # Return the singleton instance


    def _initialize(self, config_path: Optional[str] = None) -> None:
        """Initialize the ConfigManager instance. Loads and validates the configuration.

        Determines the configuration file path (using provided path or searching default locations),
        loads the configuration from the YAML file, and then validates the loaded configuration
        to ensure all required sections and settings are present.

        Args:
            config_path: Optional path to the configuration file.

        Raises:
            ConfigurationError: If configuration file loading or validation fails at any step.
        """
        self._config_path = config_path or self._find_config() # Determine config file path
        logger.info(f"Loading configuration from: {self._config_path}")
        self._load_config() # Load config from YAML
        self._validate_config() # Validate loaded config
        logger.info("Configuration loaded and validated successfully.")


    def _find_config(self) -> str:
        """Locate the configuration file by searching in standard locations.

        Searches for 'config.yaml' in the current directory, 'config/config.yaml', and the path
        specified by the environment variable 'CAG_CONFIG' (defaulting to 'config.yaml' if not set).

        Returns:
            str: The path to the configuration file if found.

        Raises:
            ConfigurationError: If no configuration file is found in any of the search paths.
        """
        search_paths: List[Path] = [
            Path("config.yaml"), # Current directory
            Path("config/config.yaml"), # 'config' subdirectory
            Path(os.getenv("CAG_CONFIG", "config.yaml")), # Environment variable, defaults to 'config.yaml'
        ]

        for path in search_paths:
            if path.is_file(): # Check if path exists and is a file
                logger.debug(f"Configuration file found at: {path}")
                return str(path) # Return path as string if found

        # If no config file found in search paths, raise ConfigurationError
        raise ConfigurationError(
            f"No configuration file found. Searched in: {', '.join(str(p) for p in search_paths)}"
        )


    def _load_config(self) -> None:
        """Load configuration data from the YAML file specified by self._config_path.

        Handles file reading and YAML parsing, including environment variable substitution
        within the YAML file.

        Raises:
            ConfigurationError: If there is an error reading the file or parsing the YAML content.
        """
        try:
            with open(self._config_path, 'r') as f:
                # Load YAML and substitute environment variables
                self._config = self._replace_env_vars(yaml.safe_load(f))
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found at path: {self._config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML parsing error in configuration file: {e}")
        except Exception as e: # Catch-all for other potential loading errors
            raise ConfigurationError(f"Failed to load configuration file: {str(e)}")


    def _replace_env_vars(self, config_data: Any) -> Any:
        """Recursively replace environment variables in the configuration data.

        Looks for strings in the format '${ENV_VAR_NAME}' in the configuration and replaces
        them with the value of the environment variable 'ENV_VAR_NAME'. If the environment
        variable is not set, it leaves the placeholder as is (or you can customize this behavior).

        Args:
            config_data: The configuration data (can be a dict, list, string, or other YAML-supported type).

        Returns:
            Any: The configuration data with environment variables substituted.
        """
        if isinstance(config_data, dict):
            return {k: self._replace_env_vars(v) for k, v in config_data.items()} # Recursively process dictionaries
        elif isinstance(config_data, list):
            return [self._replace_env_vars(item) for item in config_data] # Recursively process lists
        elif isinstance(config_data, str):
            if config_data.startswith('${') and config_data.endswith('}'): # Check for env var pattern
                env_var_name = config_data[2:-1] # Extract env var name (remove '${' and '}')
                return os.getenv(env_var_name, config_data) # Get env var value, default to original string if not set
            return config_data # Return string as is if not an env var placeholder
        return config_data # Return non-string, non-dict, non-list data as is


    def _validate_config(self) -> None:
        """Validate the loaded configuration to ensure required sections and settings are present and valid.

        Performs checks to ensure that the configuration contains all necessary sections (cache, embedding, llm, similarity)
        and that each section contains its required settings. Raises a ConfigurationError if validation fails.

        Raises:
            ConfigurationError: If any required sections or settings are missing or invalid.
        """
        required_sections: List[str] = ['cache', 'embedding', 'llm', 'similarity']
        required_settings: Dict[str, List[str]] = {
            'cache': ['type'], # 'max_size' is optional now, defaults in InMemoryCache
            'embedding': ['type'], # 'model_name' is optional, defaults in SentenceTransformersEmbeddingModel
            'llm': ['type'], # 'api_key' and 'model_endpoint' are conditionally required, checked in LLMFactory/LLM classes
            'similarity': ['metric'] # 'threshold' is optional, defaults in SimilarityMetric
        }
        valid_cache_types = ['in_memory', 'redis']
        valid_embedding_types = ['sentence_transformers', 'openai']
        valid_llm_types = ['huggingface', 'openai']
        valid_similarity_metrics = ['cosine', 'euclidean']


        # Check for required top-level sections
        missing_sections = [section for section in required_sections if section not in self._config]
        if missing_sections:
            raise ConfigurationError(f"Missing required configuration sections: {missing_sections}")

        # Validate settings within each required section
        for section_name, required_settings_list in required_settings.items():
            section_config = self._config.get(section_name, {}) # Get section config, default to empty dict if missing

            # Check for missing settings within the section
            missing_settings = [setting for setting in required_settings_list if setting not in section_config]
            if missing_settings:
                raise ConfigurationError(
                    f"Missing required settings in '{section_name}' section: {missing_settings}"
                )

            # Type-specific validation for 'cache' section
            if section_name == 'cache':
                cache_type = section_config.get('type')
                if cache_type not in valid_cache_types:
                    raise ConfigurationError(f"Invalid cache type '{cache_type}'. Must be one of: {valid_cache_types}")
                if cache_type == 'in_memory':
                    max_size = section_config.get('max_size')
                    if max_size is not None and not isinstance(max_size, int):
                        raise ConfigurationError(f"Invalid 'max_size' for in_memory cache. Must be an integer.")
                elif cache_type == 'redis':
                    redis_config = section_config.get('redis', {})
                    # You could add more detailed validation for Redis config here if needed

            # Type-specific validation for 'embedding' section
            elif section_name == 'embedding':
                embedding_type = section_config.get('type')
                if embedding_type not in valid_embedding_types:
                    raise ConfigurationError(f"Invalid embedding type '{embedding_type}'. Must be one of: {valid_embedding_types}")
                # OpenAI API key validation will be handled when creating OpenAIEmbeddingModel

            # Type-specific validation for 'llm' section
            elif section_name == 'llm':
                llm_type = section_config.get('type')
                if llm_type not in valid_llm_types:
                    raise ConfigurationError(f"Invalid LLM type '{llm_type}'. Must be one of: {valid_llm_types}")
                # API key and model_endpoint validation for HuggingFace and OpenAI will be in LLMFactory/LLM classes

            # Type-specific validation for 'similarity' section
            elif section_name == 'similarity':
                similarity_metric = section_config.get('metric')
                if similarity_metric not in valid_similarity_metrics:
                    raise ConfigurationError(f"Invalid similarity metric '{similarity_metric}'. Must be one of: {valid_similarity_metrics}")
                threshold = section_config.get('threshold')
                if threshold is not None and not isinstance(threshold, (int, float)):
                    raise ConfigurationError(f"Invalid 'threshold' for similarity. Must be a float or integer.")
                sigma = section_config.get('euclidean', {}).get('sigma') # Get sigma from 'euclidean' subsection
                if similarity_metric == 'euclidean' and sigma is not None and not isinstance(sigma, (int, float)):
                    raise ConfigurationError(f"Invalid 'sigma' for euclidean similarity. Must be a float or integer.")


    def get(self, path: str, default: Any = None) -> Any:
        """Retrieve a configuration value using a dot-separated path.

        Allows accessing nested configuration values within the YAML structure using a path string
        like 'section.subsection.setting'. If the path is not found, returns the provided default value.

        Args:
            path: Dot-separated path to the configuration value (e.g., 'cache.type', 'llm.openai.model').
            default: Optional default value to return if the path does not exist in the configuration.
                     Defaults to None.

        Returns:
            Any: The configuration value at the specified path, or the default value if the path is not found.
        """
        try:
            value = self._config # Start at the root of the config
            for key in path.split('.'): # Split path into keys
                value = value[key] # Traverse dict using keys
            return value # Return value if path is valid
        except (KeyError, TypeError): # Catch KeyError for missing keys, TypeError if trying to index non-dict
            return default # Return default if path not found


    def get_cache_config(self) -> Dict[str, Any]:
        """Retrieve the entire 'cache' section of the configuration.

        Returns:
            Dict[str, Any]: Dictionary containing the configuration settings for the cache.
                             Returns an empty dictionary if the 'cache' section is not found (though validation should prevent this).
        """
        return self._config.get('cache', {})


    def get_embedding_config(self) -> Dict[str, Any]:
        """Retrieve the entire 'embedding' section of the configuration.

        Returns:
            Dict[str, Any]: Dictionary containing the configuration settings for the embedding model.
                             Returns an empty dictionary if the 'embedding' section is not found.
        """
        return self._config.get('embedding', {})


    def get_llm_config(self) -> Dict[str, Any]:
        """Retrieve the entire 'llm' section of the configuration.

        Returns:
            Dict[str, Any]: Dictionary containing the configuration settings for the LLM.
                             Returns an empty dictionary if the 'llm' section is not found.
        """
        return self._config.get('llm', {})


    def get_similarity_config(self) -> Dict[str, Any]:
        """Retrieve the entire 'similarity' section of the configuration.

        Returns:
            Dict[str, Any]: Dictionary containing the configuration settings for similarity metrics.
                             Returns an empty dictionary if the 'similarity' section is not found.
        """
        return self._config.get('similarity', {})


    def reload(self) -> None:
        """Reload the configuration from the configuration file.

        This method reloads the configuration from the file path that was initially used
        to load the configuration. It is useful for refreshing the configuration without
        restarting the application, especially if the configuration file has been updated.

        Raises:
            ConfigurationError: If reloading the configuration fails (e.g., file not found, invalid YAML).
        """
        logger.info("Reloading configuration...")
        self._load_config() # Load config again from file
        self._validate_config() # Re-validate the reloaded config
        logger.info("Configuration reloaded successfully.")