import json
import os

from typing import List, NamedTuple, Tuple, overload
from collections import ChainMap
from deep_chainmap import DeepChainMap
from loguru import logger
from pathlib import Path
from random import randint


class TrainerConfig(NamedTuple):
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
    per_gpu_train_batch_size: int
        Batch size for each GPU.
    train_dataloader_num_workers: int
       Number of workers to preprocess & load the training dataset.
    val_dataloader_num_workers: int
       Number of workers to preprocess & load the validation dataset.
    val_batch_size: int
       Batch size for the validation dataset.
    collate_method_name: str
       Collate method to tokenize & load the data.
    collate_method_init_args: dict
       Initialization arguments to the collate method.
    train_dataloader_memoize: bool
        Whether to tokenize beforehand and save into CPU memory.

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

    Optimizer Configs:
    ------------------
    learning_rate: float
        The learning rate passed to the optimizer. Defaults to 0.0001
    adam_epsilon: float
        Adam Optimizer Epsilon. Defaults to 1e-8.

    Learning Rate Scheduler Configs:
    -----------------------------
    num_warmup_steps: int
        Number of warmup steps (low learning rate before / at the beginning of the training). Defaults to 0.
    num_hard_restart_cycles: int = -1
        Number of hard restarts of scheduling the learning rate. Defaults to -1 and use linear schedule with no hard resets.
    weight_decay: float = 0.0
        Weight decay to apply. Defaults to 0.0

    Gradient Configs:
    -----------------
    gradient_accumulation_steps: int
        Num of batches to go through before back propagation. Defaults to 1.
    max_grad_norm: float
        Parameter to clip_grad_norm. Max norm of the gradient. Defaults to 1.0


    Continue Training Configs
    ---------------------------
    continue_training: bool
        Defaults to False. Whether to continue training or not.
    continue_training_model_file_path: str
        Path to continue training a model.

    Training Common Configs:
    -----------------------
    random_seed: int
        The random seed to use (for numpy, torch, python)
    max_epochs_to_train: int
        Number of epochs to train

    Val Configs:
    ------------------------------
    val: bool
        Determine whether to evaluate on validation set. Defaults to True.
    val_percentage: float
        Percentage of the training dataset to use as validation.
    val_ratio_tolerance: float
        Ratio tolerance for splitting training dataset to train and val.
    val_per_x_epochs: int
        Defaults to 1. Validate the model every `x` epochs.
        Increase this when there are many validation metrics to compute and training time is a concern.
    val_metric_methods: List[Tuple[str, dict]]
        List of [(val_metric_method, val_metric_init_args)]

    Model Save Configs:
    ---------------------
    save_model: bool
        Determine whether to save the model or not. Defaults to False.
    save_per_x_epoch: int
        Save model every `x` epoch. Defaults to 1.
    num_models_to_save: int
        Number of models to save. Defaults to 1.
    model_save_prepend_name: str
        Prepend name of the model to save. We add `_{epoch_num}` after the prepend name.
        If left at "", we use the `model_name`.
    save_folder_path: str
        Path to the saving folder. Defaults to "checkpoints".
    save_best_model_metric: str
        Metric to determine the best model. Saves if a best model is found. Will not start saving till `min_epochs_to_train`.
    best_model_save_name: str
        Save name of the best model.If left at "", it will be named `model_name`.

    Early Stop Configs:
    --------------------
    early_stopping_strategies: List[Tuple(str, dict)] = []
        List of (early_stopping_strategy, early_stopping_strategy_init_args)

    Tensorboard Summary Writer Configs:
    -----------------------------------
    tb_log_dir: Optional[str]
        Directory to log metrics to tensorboard. Defaults to None => original behavior of summary writer.
    log_per_x_epochs : int
        Defaults to 5. Log every `x` epochs have finished.

    '''
    # Data Pipeline Config
    data_retrieval_method: str = None
    data_retrieval_init_args: dict = {}
    data_transformations: List[Tuple[str, dict]] = [],
    data_output_method: str = None,
    data_output_init_args: dict =  None,

    # DataLoader Configs
    per_gpu_train_batch_size: int = 16
    train_dataloader_num_workers: int = 8
    val_dataloader_num_workers: int = 8
    val_batch_size: int = 512
    collate_method_name: str = ""
    collate_method_init_args: dict = {}
    train_dataloader_memoize: bool = False

    # Model Init Configs
    model_classification_type: str = "ToxicTokenClassification"
    model_class: str = "ToxicTokenClassification"
    model_class_init_args: dict = {}


    # Optimizer Configs
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8

    # Learning Rate Scheduler Configs
    num_warmup_steps: int = 0
    num_hard_restart_cycles: int = -1

    # Gradient Configs
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Continue Training Configs
    continue_training: bool = True
    continue_training_model_file_path: str = ""

    # Training Common Configs
    random_seed: int = -1
    max_epochs_to_train: int = 50

    # Loss Function
    weighted_loss : bool = False

    # Val Configs
    val: bool = True
    val_percentage: float = 0.2
    val_ratio_tolerance : float = 0.05
    val_per_x_epochs : int = 1
    val_metric_methods: List[Tuple[str, dict]] = []

    # Model Save Configs
    save_model: bool = False
    save_per_x_epoch: int = 1
    num_models_to_save: int = 1
    model_save_prepend_name: str = ""
    save_folder_path: str = "checkpoints"
    save_best_model_metric : str = "FirstToxicToken_weighted avg_f1-score"
    best_model_save_name: str = ""

    # Early Stop Configs
    early_stopping_strategy: str = ""
    metric_to_compare: str = ""
    min_epochs_to_train: int = 0
    patience: int = 0

    # Tensorboard Summary Writer Configs
    log_per_x_epochs : int = 5
    tb_log_dir: str = None

    @staticmethod
    def from_config_file(config_file: str):
        '''
        Initializes the TrainerConfig from a json config file.

        Parameter:
        -----------
        config_file: str
            Path to the json config file.
        '''
        with open(config_file) as f:
            config = json.load(f)
        return TrainerConfig.from_config(config)

    @staticmethod
    def from_config_files(config_files: List[str]):
        '''
        Initializes the TrainerConfig from multiple json config files.
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
        return TrainerConfig.from_configs(configs)


    @staticmethod
    def from_config(config: dict):
        '''
        Initializes the Trainer Config from a dictionary that possibly contains more info than necessary.

        Parameter:
        ----------
        config: dict
            Dictionary containing appropriate key-value pairs.
        '''
        config_fields = TrainerConfig._fields

        filtered_config = {}
        for config_field in config_fields:
            if (config_field in config):
                filtered_config[config_field] = config[config_field]

        # Change the random seed to a random number between 0 and 99999 if not set.
        if ("random_seed" not in filtered_config):
            filtered_config["random_seed"] = randint(0, 9999)
            logger.info(f"Set random seed to {filtered_config['random_seed']}")

        return TrainerConfig(**filtered_config)

    @staticmethod
    def from_configs(configs: List[dict]):
        '''
        Initializes the Trainer Config from a dictionary that possibly contains more info than necessary.

        Parameter:
        ----------
        config: dict
            Dictionary containing appropriate key-value pairs.
        '''
        final_config = DeepChainMap(*configs).to_dict()
        return TrainerConfig.from_config(final_config)


    @staticmethod
    def from_configs_and_config_files(configs: List[dict], config_files: List[str]):
        '''
        Initializes the Trainer Config from a dictionary that possibly contains more info than necessary.

        Parameter:
        ----------
        config: dict
            Dictionary containing appropriate key-value pairs.
        '''
        for config_file in config_files:
            with open(config_file) as f:
                configs.append(json.load(f))
        return TrainerConfig.from_configs(configs)

    def save(self, output_file: str):
        '''
        Given the TrainerConfig, saves appropriate information into a JSON file.

        Parameter:
        -----------
        output_file: str
            place to save the config.
        '''
        configs = self._asdict()

        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok = True)

        with open(output_file, 'w') as json_file:
            json.dump(configs, json_file, indent=1)
