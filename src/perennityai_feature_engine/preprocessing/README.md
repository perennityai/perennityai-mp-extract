# Preprocessor

The `Preprocessor` class provides a flexible and configurable pipeline for preprocessing data using various processors. It is designed to work with different data types and configurations, delegating tasks to an internal `FeatureProcessor` for specific data processing.

## Table of Contents

- [Installation](#installation)
- [Class Overview](#class-overview)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Loading from Pretrained Configuration](#loading-from-pretrained-configuration)
  - [Data Preprocessing](#data-preprocessing)
  - [Dataset Preprocessing](#dataset-preprocessing)
- [Configuration Options](#configuration-options)
- [License](#license)

---

## Installation

To use the `Preprocessor` class, ensure that you have all dependencies installed. The `FeatureProcessor` module, along with the necessary configuration files and dependencies, should be part of your project setup.

---

## Class Overview

The `Preprocessor` class initializes with an optional configuration, either directly as a dictionary or loaded from a pretrained configuration file. It offers methods to preprocess individual data items as well as an entire dataset.

### Key Methods

- `__init__(config=None)`: Initializes the `Preprocessor` with a specified or default configuration.
- `from_pretrained(path: str)`: Loads a `Preprocessor` instance using a configuration file from a given path.
- `preprocess_data(data)`: Preprocesses individual data items based on configuration.
- `preprocess_dataset(padding=False, extract_only=False)`: Preprocesses the entire dataset, optionally with padding or extraction-only options.

---

## Usage

### Initialization

You can initialize the `Preprocessor` class with a configuration dictionary specifying settings such as the processor type, input and output directories, and processing options.

```python
from preprocessor import Preprocessor

# Example configuration dictionary
config = {
    "processor": "dynamic",
    "input_dir": "data/raw",
    "output_dir": "data/processed",
    "extract": True,
    "average_out_face": True,
    "overwrite": False
}

# Initialize Preprocessor with custom configuration
preprocessor = Preprocessor(config=config)
```


# Loading from Pretrained Configuration
To use a pretrained configuration, load it from a config.json file in the specified path. This file should contain the necessary configuration parameters in JSON format.

# Load Preprocessor from a pretrained configuration
```
preprocessor = Preprocessor.from_pretrained("path/to/config_directory")
```

# Data Preprocessing
To preprocess a single data item, use the preprocess_data method. This method takes in the data and processes it according to the configuration.

```python
data = ...  # Load your data here
preprocessed_data = preprocessor.preprocess_data(data)
```

To preprocess an entire dataset, use the preprocess_dataset method. You can optionally set padding and extract_only parameters to customize the dataset processing.

# Preprocess the entire dataset
```python
preprocessor.preprocess_dataset(padding=True, extract_only=False)
```

# Configuration Options
The Preprocessor class supports the following configuration options:

- processor: Specifies the processor type, e.g., "dynamic", "static".
- input_dir: Directory containing input data.
- output_dir: Directory where processed data will be saved.
- extract: Boolean indicating whether to extract features during processing.
- average_out_face: Boolean indicating whether to average out face data.
- overwrite: Boolean indicating whether to overwrite existing processed data.

# License
This project is licensed under the MIT License - see the LICENSE file for details.


Copy this text into a `README.md` file to give a structured and easy-to-read documentation for the `Preprocessor` class. This format will render well on GitHub and other Markdown viewers.


