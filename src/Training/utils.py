from loguru import logger
from prettytable import PrettyTable
from typing import Dict, Tuple, Union

import numpy as np
import random
import torch

def gpu_information_summary(show: bool = True) -> Tuple[int, torch.device]:
    """
    :param show: Controls whether or not to print the summary information
    :return: number of gpus and the device (CPU or GPU)
    """
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name() if n_gpu > 0 else "None"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    table = PrettyTable()
    table.field_names = ["Key", "Value"]
    table.add_row(["GPU", gpu_name])
    table.add_row(["Number of GPUs", n_gpu])
    if show:
        logger.info(table)
    return n_gpu, device


def set_random_seed(seed_value: int, n_gpu: int) -> None:
    '''
    Sets the random seed for libraries used: random, np and torch
    '''
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_value)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    '''
    Detaches the tensor, move to cpu and then convert to numpy
    '''
    return tensor.detach().cpu().numpy()
