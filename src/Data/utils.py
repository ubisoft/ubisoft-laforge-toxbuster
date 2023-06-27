from loguru import logger

from src.Data.Dataset import ToxDataset
from src.Data import Collate

def get_num_of_classes(training_dataset: ToxDataset, val_dataset: ToxDataset):
    '''
    Gets the number of classes in training dataset.
    Also does a sanity check on:
    * classes are consecutive
    * num of clases in train and val are equivalent

    Parameter:
    ------------
    training_dataset: ToxDataset
        Training dataset
    val_dataset: ToxDataset
        Validation dataset
    '''
    train_label_value_counts = training_dataset.label_value_counts
    train_num_of_classes = max(train_label_value_counts.keys()) + 1
    ToxDataset.check_consecutive_classes(train_label_value_counts, train_num_of_classes, "train")


    if (val_dataset):
        val_label_value_counts = val_dataset.label_value_counts
        val_num_of_classes = max(val_label_value_counts.keys()) + 1
        ToxDataset.check_consecutive_classes(val_label_value_counts, val_num_of_classes, "validation")

        if (train_num_of_classes != val_num_of_classes):
            logger.error(f"Train dataset has {train_num_of_classes} classes while validation has {val_num_of_classes} classes.")
    logger.debug(f"Number of classes to be trained: {train_num_of_classes}")
    return train_num_of_classes


@logger.catch("Could not instantiate the collate function")
def collate_init(collate_method_name: str,
                 collate_method_init_args: dict):
    '''
    Instantiates the collate to pass to the data loader.
    '''
    collate = getattr(Collate, collate_method_name)
    logger.info(f"Instantiating {collate_method_name} with the following args:\n {collate_method_init_args}")
    return collate(**collate_method_init_args)

