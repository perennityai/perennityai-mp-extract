import json
import os
import os
from .feature_processor import FeatureProcessor

class Preprocessor:
    def __init__(self, config=None):
        """
        Initializes the DataPreprocessor with optional configuration.

        Args:
            config (dict, optional): A dictionary with configuration parameters. Default is None.
        """
        # Initialize with default values or provided config
        self.config = config or {}
        self.processor = self.config.get("processor", "dynamic")
        self.padding = self.config.get("padding", False)
        
        # Initialize FeatureProcessor
        self.__feature_processor = FeatureProcessor(processor=self.processor,
                                                    input_dir = self.config.get("processor", "input_dir"),
                                                    output_dir = self.config.get("processor", "output_dir"),
                                                    extract = self.config.get("extract", False),
                                                    average_out_face = self.config.get("average_out_face", False),
                                                    overwrite = self.config.get("overwrite", False),
                                                    pid = self.config.get("pid", 0)
                                                  )

    @classmethod
    def from_pretrained(cls, config):
        """
        Loads a DataPreprocessor instance with pretrained configuration.

        Args:
            config (str or dict): The path to the pretrained configuration file or directory or JSON file.

        Returns:
            DataPreprocessor: An instance of DataPreprocessor initialized with the pretrained configuration.
        """
        
        if not not isinstance(config, str) or not isinstance(config, dict):
            raise FileNotFoundError(f"No configuration file found at {config}")
        
        # Load configuration from a file (e.g., JSON format)
        if not isinstance(config, dict):
            if '.json' not in config_path:
                config_path = os.path.join(config, "config.json")
            else:
                config_path = config

            with open(config_path, "r") as f:
                config = json.load(f)

        # Instantiate DataPreprocessor with the loaded configuration
        return cls(config=config)
    

    @classmethod
    def from_pretrained(cls, config):
        """
        Loads a pretrained configuration for initializing a DataVisualizer instance.

        Args:
            config_path (str): Path to the JSON file containing the configuration.

        Returns:
            DataVisualizer: An instance initialized with pretrained configurations.
        """
        if not os.path.isfile(config) or not isinstance(config, dict):
            raise FileNotFoundError(f"No configuration file found at {config}")

        if not isinstance(config, dict):
            with open(config, "r") as file:
                config = json.load(file)
        
        return cls(
            input_file=config.get('input_file', ''),
            input_dir=config.get('input_dir', ''),
            output_dir=config.get('output_dir', ''),
            data_input_format=config.get('data_input_format', 'csv'),
            encoding=config.get('encoding','ISO-8859-1'),
            verbose=config.get('verbose','INFO')
        )


    def preprocess_data(self, data):
        """
        Preprocesses the data using the FeatureProcessor based on the configuration.

        Args:
            data (any): The input data to preprocess.

        Returns:
            any: The preprocessed data.
        """
        # Delegate preprocessing to FeatureProcessor
        return self.__feature_processor.extract_frame_features(
            data=data,
            processor=self.processor,
            padding=self.padding,
            average_out_face=self.average_out_face
        )

    def preprocess_dataset(self, padding = False, extract_only = False):
        """
        Preprocesses the dataset using the FeatureProcessor based on the configuration.

        Args:
            data (any): The input data to preprocess.

        Returns:
            None: The preprocessed dataset to filesystem.
        """
        # Delegate preprocessing to FeatureProcessor
        self.__feature_processor.process_dataset(
            padding=padding,
            extract_only=extract_only,
        )
    