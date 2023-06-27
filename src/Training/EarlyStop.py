from abc import ABC, abstractmethod
from typing import Dict, Union

class EarlyStoppingStrategy(ABC):
    '''
    Strategy to early stop in training
    '''
    def __init__(self,
                 min_epochs_to_train: int,
                 metric_to_compare: str,
                 patience: int):
        '''
        Initializes the early stopping strategy.

        Parameter:
        ----------
        min_epochs_to_train: int
            The minimum number of epochs to train before early stopping logic is tested.
        metric_to_compare: str
            String name of the validation metric to compare => Should correlate to the one saved in TensorBoard.
        patience: int
            Number of epochs before no improvement in `metric_to_compare`.
        '''
        self.patience = max(1, patience)
        self.metric_to_compare = metric_to_compare
        self.min_epochs_to_train = max(0, min_epochs_to_train)

    @abstractmethod
    def __call__(self, epoch: int,
                 val_metrics: Dict[str, Union[float, Dict[str, float]]]) -> bool:
        '''
        Determines whether to early stop or not.

        Parameter:
        ----------
        epoch: epoch
            The number of the current epoch.
        val_metrics: Dict[str, Union[float, Dict[str, float]]]
            The validation metrics collected during the validation loop.
        '''
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class BestValMetric(EarlyStoppingStrategy):
    '''
    Strategy to early stop in training
    '''
    def __init__(self,
                 min_epochs_to_train: int,
                 metric_to_compare: str,
                 patience: int):
        '''
        Initializes the early stopping strategy.

        Parameter:
        ----------
        min_epochs_to_train: int
            The minimum number of epochs to train before early stopping logic is tested.
        metric_to_compare: str
            String name of the validation metric to compare. Should be either 'loss' or metrics calculated in SentenceLevelMetrics.
            Name would be {SentenceLevelMetric}_{metric_name}.
        patience: int
            Number of epochs before no improvement in `metric_to_compare`.
        '''
        super().__init__(min_epochs_to_train, metric_to_compare, patience)

        self.epochs_with_no_improvement = 0
        self.best_val_metric = -1


    def __call__(self, epoch: int,
                 val_metrics: Dict[str, float]) -> bool:
        '''
        Determines whether to early stop or not.

        Parameter:
        ----------
        epoch: int
            The current epoch number.
        val_metrics: Dict[str, Union[float, Dict[str, float]]]
            The validation metrics collected during the validation loop.
        '''
        # Get metric we want to compare
        new_metric = val_metrics[self.metric_to_compare]

        if new_metric > self.best_val_metric:
            self.best_val_metric = new_metric
            self.epochs_with_no_improvement = 0
        else:
            self.epochs_with_no_improvement += 1

        return epoch > self.min_epochs_to_train and self.epochs_with_no_improvement >= self.patience

    def __str__(self):
        return f"BestValMetric on `{self.metric_to_compare}` with min training epoch of `{self.min_epochs_to_train}` and patience of `{self.patience}`"

class NoEarlyStopping(EarlyStoppingStrategy):
    '''
    Strategy to early stop in training
    '''
    def __init__(self,
                 min_epochs_to_train: int = 0,
                 metric_to_compare: str = "",
                 patience: int = 1):
        '''
        Initializes the early stopping strategy.

        Parameter:
        ----------
        min_epochs_to_train: int
            The minimum number of epochs to train before early stopping logic is tested.
        metric_to_compare: str
            String name of the validation metric to compare => Should correlate to the one saved in TensorBoard.
        patience: int
            Number of epochs before no improvement in `metric_to_compare`.
        '''
        super().__init__(min_epochs_to_train, metric_to_compare, patience)


    def __call__(self, epoch_num: int, val_metrics: Dict[str, Union[float, Dict[str, float]]]) -> bool:
        '''
        Determines whether to early stop or not.

        Parameter:
        ----------
        epoch_num: int
            The current epoch number.
        val_metrics: Dict[str, Union[float, Dict[str, float]]]
            The validation metrics collected during the validation loop.
        '''
        return False


    def __str__(self):
        return "NoEarlyStopping"


