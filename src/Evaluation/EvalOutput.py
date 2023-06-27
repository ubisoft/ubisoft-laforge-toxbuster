from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import pickle


@dataclass
class EvalOutput:
    '''
    Container for one sample from our dataset.

    Attributes:
    ----------
    predicted_labels: list[list[int]]
        Predicted labels from the dataset.
    predicted_labels_confidences: list[list[int]]
        The confidence level for each predicted token from the dataset.
    predicted_labels_confidences: list[list[list[int]]]
        The confidence level of each class for each predicted token from the dataset.
    true_labels: list[list[int]]
        True labels from the dataset. If we are inferring, the true labels will be a label mask instead.
    loss: float
        Loss of the dataset. If inferring, the loss would be meaningless.
    '''
    predicted_labels: List[List[int]]
    predicted_labels_confidence: List[List[int]]
    predicted_labels_confidences: List[List[int]]
    true_labels: List[List[int]]
    loss: float

def update_df(output: EvalOutput,
              df: pd.DataFrame,
              include_confidence_levels: bool = True,
              include_per_class_confidence_levels: bool = True,
              include_true_labels: bool = True):
    '''
    Adds the eval output results to the DataFrame
    ***NEW COLUMNS WILL BE ADDED***

    Parameter:
    -----------
    output: EvalOutput
        The evaluation output
    df: pd.DataFrame
        The input text dataframe
    include_confidence_levels: bool
        Defaults to True. Whether to save the confidence level per token.
    include_per_class_confidence_levels: bool
        Defaults to True. Whether to save the confidence level per token per class.
    include_true_labels: bool
        Defaults to True. Whether to save the true labels or not.
    '''
    df["predictions"] = output.predicted_labels
    df["predictions"] = df["predictions"].apply(lambda x: list(x))

    if (include_confidence_levels):
        df["confidence_levels"] = output.predicted_labels_confidence
        df["confidence_levels"] = df["confidence_levels"].apply(lambda x: list(x))

    if (include_per_class_confidence_levels):
        df["per_class_confidence_levels"] = output.predicted_labels_confidences
        df["per_class_confidence_levels"] = df["per_class_confidence_levels"].apply(lambda x: [list(i) for i in x])

    if (include_true_labels):
        df["true_labels"] = output.true_labels

    return df

def update_and_save_to_csv(output: EvalOutput,
                           df: pd.DataFrame,
                           save_folder_name: str,
                           save_file_name: str,
                           include_confidence_levels: bool = True,
                           include_per_class_confidence_levels: bool = True,
                           include_true_labels: bool = True):
    '''
    Adds the eval output results to the DataFrame and saves the result into a csv
    ***NEW COLUMNS WILL BE ADDED***

    Parameter:
    -----------
    output: EvalOutput
        The evaluation output
    df: pd.DataFrame
        The input text dataframe
    save_folder_name: str
        The folder to save
    save_file_name: str
        The file name to save
    include_confidence_levels: bool
        Defaults to True. Whether to save the confidence level per token.
    include_per_class_confidence_levels: bool
        Defaults to True. Whether to save the confidence level per token per class.
    include_true_labels: bool
        Defaults to True. Whether to save the true labels or not.

    '''
    df = update_df(output, df,
                   include_confidence_levels,
                   include_per_class_confidence_levels,
                   include_true_labels)

    file_name = f"{save_folder_name}/{save_file_name}"
    Path(save_folder_name).mkdir(parents=True, exist_ok=True)
    #df.to_csv(file_name, index=False, sep=";")

    with open(file_name, 'wb') as f:
        pickle.dump(df, f)
    return df

