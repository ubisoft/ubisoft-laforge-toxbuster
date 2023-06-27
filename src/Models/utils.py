from importlib import import_module
from loguru import logger
from pathlib import Path
from typing import Optional

import torch

def save_checkpoint(epoch: int,
                    model: torch.nn.Module,
                    model_checkpoint_name: str,
                    optimizer: torch.optim.Optimizer,
                    best_model_metric_name: str,
                    best_model_metric: float,
                    save_folder_path: str,
                    ):
    '''
    Checkpoint that saves the model state, optimizer state and number of epochs trained.

    Parameter:
    ----------
    epoch: int
        Number of epochs trained for this model.
    model: torch.nn.Module
        Model to be saved.
    model_checkpoint_name: str
        Filepath of the model to be saved.
    optimizer: torch.optim.Optimizer
        Optimizer's state to be saved.
    best_model_metric_name: str
        Name of the best model metric.
    best_model_metric: float
        Value of the best model.
    save_folder_path: str
        Folder where the checkpoint will be saved.
    '''

    # Ensure path is available.
    Path(save_folder_path).mkdir(parents=True, exist_ok=True)

    torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                best_model_metric_name: best_model_metric
                },
                model_checkpoint_name)

def load_checkpoint(model: torch.nn.Module,
                    checkpoint_path: str,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    best_model_metric_name: str = ""
                    ):
    '''
    Loads the model based on the saved checkpoint.

    Parameter:
    ----------
    model: torch.nn.Module
        Initial model
    checkpoint_path: str
        Path to the checkpoint
    optimizer: Optional[torch.optim.Optimizer]
        Defaults to none. Loads the saved optimizer state if optimizer is provided.
    best_model_metric_name: str
        Name of the best model metric. Defaults to "".


    Returns:
    --------
    (model, optimizer, epoch, best_model_metric) where
        - Updated model with saved state
        - Updated optimizer with saved state / None
        - Number of epochs already trained / -1
        - metric of the best_model / -100
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # check if state_dict has "module" prepended
    model_state_dict = unwrap_module_in_state_dict_if_exist(checkpoint["model_state_dict"])
    model.load_state_dict(model_state_dict)

    epoch = checkpoint["epoch"] if 'epoch' in checkpoint else -1
    best_model_metric = checkpoint[best_model_metric_name] if best_model_metric_name in checkpoint else -100

    if (optimizer is not None):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, epoch, best_model_metric

def unwrap_module_in_state_dict_if_exist(state_dict):
    '''
    Loops through all keys in state_dict. If "module" prepends all of them, we create a new state_dict without it.
    '''
    new_state_dict = {}
    for i in state_dict.keys():
        if not i.startswith("module"):
            return state_dict
        else:
            new_state_dict[i[7:]] = state_dict[i]
    return new_state_dict


def model_init(model_classification_type: str,
               model_class: str,
               model_class_init_args: dict) -> torch.nn.Module:
    '''
    Wrapper function to initialize a model.

    Parameter:
    -----------
    model_classification_type: str
        Type of classification; corresponds to the module name the model resides in
    model_class: str
        Str name of the model.
    model_class_init_args
    '''

    model_classification_type = import_module(f"src.Models.{model_classification_type}")
    model_class = getattr(model_classification_type, model_class)
    model = model_class(**model_class_init_args)
    return model