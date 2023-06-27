import json
import os
from pathlib import Path

from typing import List, NamedTuple, Tuple
from deep_chainmap import DeepChainMap


class InfererConfig(NamedTuple):
    '''
    All configurations related to the trainer.

    Data Pipeline Config:
    ----------------------
    data_retrieval_method: str
        Method to retrieve the data. Look @ data_retrieval class
    data_retrieval_init_args: dict
        Arguments to pass to initialize the data retrieval method
    data_transformations: list[Tuple[str, dict]]
        List of (data_transformation_method, data_transformation_init_args)
    data_output_method: str
        Method to output the data
    data_output_init_args: dict
        Arguments to pass to initialize the data output method.

    DataLoader Configs:
    -------------------
    val_dataloader_num_workers: int
       Number of workers to preprocess & load the validation dataset.
    val_batch_size: int
       Batch size for the validation dataset.
    collate_method_name: str
       Collate method to tokenize & load the data.
    collate_method_init_args: dict
       Initialization arguments to the collate method.

    Model Init Configs:
    --------------------
    model_type: str
       Type of model.
    model_init_args:
       Any arguments need to initialize the model.
    model_name: str
        Name of the base-model from HuggingFace.
    num_classes: int
        Number of classes to classify.


    Model Checkpoint
    ---------------------------
    continue_training_model_file_path: str
        Path to continue training a model.
    save_folder_path: str
        Path to save infer results. Defaults to "Infer"

    '''
    # Data Pipeline Config
    data_retrieval_method: str = None
    data_retrieval_init_args: dict = {}
    data_transformations: List[Tuple[str, dict]] = [],
    data_output_method: str = None,
    data_output_init_args: dict = None,

    # DataLoader Configs
    val_dataloader_num_workers: int = 8
    val_batch_size: int = 512
    collate_method_name: str = ""
    collate_method_init_args: dict = {}

    # Model Init Configs
    model_classification_type: str = "ToxicTokenClassification"
    model_class: str = "ToxicTokenClassification"
    model_class_init_args: dict = {}
    num_classes: int = -1

    # Model Checkpoint
    checkpoint_path: str = ""

    # Save Folder Path
    save_folder_path: str = ""
    predicted_labels_save_file_name: str = ""

    # Output info
    confidence_levels:bool = False
    per_class_confidence_levels:bool = False


    @staticmethod
    def from_config_file(config_file: str):
        '''
        Initializes the InfererConfig from a json config file.

        Parameter:
        -----------
        config_file: str
            Path to the json config file.
        '''
        with open(config_file) as f:
            config = json.load(f)
        return InfererConfig.from_config(config)

    @staticmethod
    def from_config_files(config_files: List[str]):
        '''
        Initializes the InfererConfig from multiple json config files.
        If there are repeated keys in separate config files, it uses the first seen key.

        A typical example use case would be [experiment_config_file, base_config_file]

        Parameter:
        -----------
        config_files: List[str]
            A list of file paths corresponding to the config files.
        '''
        configs = []
        for config_file in config_files:
            with open(config_file) as f:
                configs.append(json.load(f))
        return InfererConfig.from_configs(configs)

    @staticmethod
    def from_config(config: dict):
        '''
        Initializes the Inferer Config from a dictionary that possibly contains more info than necessary.

        Parameter:
        ----------
        config: dict
            Dictionary containing appropriate key-value pairs.
        '''
        config_fields = InfererConfig._fields

        filtered_config = {}
        for config_field in config_fields:
            if (config_field in config):
                filtered_config[config_field] = config[config_field]

        return InfererConfig(**filtered_config)

    @staticmethod
    def from_configs(configs: List[dict]):
        '''
        Initializes the Inferer Config from a dictionary that possibly contains more info than necessary.

        Parameter:
        ----------
        config: dict
            Dictionary containing appropriate key-value pairs.
        '''
        final_config = DeepChainMap(*configs).to_dict()
        return InfererConfig.from_config(final_config)

    @staticmethod
    def from_configs_and_config_files(configs: List[dict], config_files: List[str]):
        '''
        Initializes the Inferer Config from a dictionary that possibly contains more info than necessary.

        Parameter:
        ----------
        config: dict
            Dictionary containing appropriate key-value pairs.
        '''
        for config_file in config_files:
            with open(config_file) as f:
                configs.append(json.load(f))
        return InfererConfig.from_configs(configs)

    def save(self, output_file: str):
        '''
        Given the InfererConfig, saves appropriate information into a JSON file.

        Parameter:
        -----------
        output_file: str
            place to save the config.
        '''
        configs = self._asdict()

        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as json_file:
            json.dump(configs, json_file, indent=1)
