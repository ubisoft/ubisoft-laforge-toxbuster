from collections import Counter
from dataclasses import dataclass
from loguru import logger
from torch.utils.data import Dataset
from typing import List, Optional
from prettytable import PrettyTable

import json
import os
import pandas as pd
import numpy as np


@dataclass
class ToxDataReturn:
    '''
    Container for one sample from our dataset.

    Attributes:
    ----------
    context: List[str]
        Our context is currently a list of words.
    text: List[str]
        Our current line of chat.
    label: Optional[List[int]]
        The label corresponding to each word in `text`.
        Must be in the same order as `text`.
    context_label: Optional[List[int]]
        The label corresponding to each word in the `context`. Must be in the same order as `context`.
    chat_type: Optional[List[int]]
        The chat type corresponding to each word in `text`.
    context_chat_types: Optional[List[int]]
        The chat type corresponding to each word in `context`.
    player_id: Optional[List[int]]
        The player_id corresponding to each word written by the player in `text`.
    context_player_ids: Optional[List[int]]
        The player_id corresponding to each word written by players in `context`.
    team_id: Optional[List[int]]
        The team_id the player belongs to corresponding to each word written by said player in `text`.
    context_team_ids: Optional[List[int]]
        The team_id of player for each corresponding word in `context`.

    '''
    context: List[str]
    text: List[str]
    label: Optional[List[int]] = None
    context_label: Optional[List[int]] = None

    chat_type: Optional[List[int]] = None
    player_id: Optional[List[int]] = None
    team_id: Optional[List[int]] = None
    context_chat_types: Optional[List[int]] = None
    context_player_ids: Optional[List[int]] = None
    context_team_ids: Optional[List[int]] = None

class ToxDataset(Dataset):
    '''
    Our toxicity dataset. Each sample is a `ToxDataReturn`.
    '''

    def __init__(self, data: pd.DataFrame,
                       train: bool = True,
                       ):
        '''
        Initializes a ToxDataSet.

        To include speaker_segmentation, the dataset must also include the following columns:
            chat_type, player_id, team_id
            context_chat_types, context_player_ids, context_team_ids

        Parameters:
        -----------
        data: pd.DataFrame
            The dataset. It must contain the following columns:
            ["context", "full_text", "matchid", "line_index", "label"]
        train: bool
            Whether this dataset is for train / not.
        '''
        self.data = data
        self.train = train
        self.label_value_counts = self.__get_label_value_counts()

    @staticmethod
    def from_file(data_path: str, train: bool = True):
        '''
        Static constuctor to load the data from a file. The label column is
        only needed if this is a training dataset.

        File Structure:
        --------------
        Header Row: Must contain `full_text`, `context`, `label`.
        Separator: `;`
        Sample Row:`['gr']` ; `['', 'what', 'league', 'do', 'u', 'announce', 'for']` ; `[0]`

        Returns:
        -------
        ToxDataset with data read from provided file.
        '''
        data_path = os.path.expanduser(data_path)
        df = pd.read_csv(data_path, sep=";")

        # Fill the empty rows to the empty str list
        # Lambda function:convert the stored str list of words into list of str
        df["context"] = df["context"].fillna("[]").apply(lambda x: eval(x))
        df["full_text"] = df["full_text"].fillna("[]").apply(lambda x: eval(x))

        if train:
            df["label"] = df["label"].fillna("[]").apply(lambda x: eval(x))

        return ToxDataset(df, train)

    def __len__(self) -> int:
        ''' Size of our dataset'''
        return len(self.data)

    def __getitem__(self, item) -> ToxDataReturn:
        '''One row in our dataset'''
        sample = self.data.iloc[item]

        return ToxDataReturn(
                text=sample["full_text"],
                label=sample["label"] if self.train else None,
                chat_type = sample["chat_type"] if 'chat_type' in sample else None,
                player_id = sample["player_id"] if 'player_id' in sample else None,
                team_id = sample["team_id"] if 'team_id' in sample else None,
                context=sample["context"],
                context_label=sample["context_label"] if "context_label" in sample else None,
                context_chat_types = sample["context_chat_types"] if "context_chat_types" in sample else None,
                context_player_ids = sample["context_player_ids"] if "context_player_ids" in sample else None,
                context_team_ids = sample["context_team_ids"] if 'context_team_ids' in sample else None,
            )

    def __get_label_value_counts(self)-> Counter:
        '''
        Retrieve a counter on the the label
        '''
        if (not self.train):
            return Counter()

        all_labels_flattened = []
        for instance_labels in self.data["label"]:
            for label in instance_labels:
                all_labels_flattened.append(label)
        return Counter(all_labels_flattened)

    @staticmethod
    def check_consecutive_classes(label_value_counts: Counter, number_of_classes: int, set_type: str):
        '''
        Checks if the classes are consecutive.

        Parameter:
        -------------
        label_value_counts: Counter
            Counter containing the unique label and corresponding counts.
        number_of_classes: int
            The expected number of classes
        set_type: str
            The type of dataset this is. Used in logging.
        '''
        missing_classes = []
        for class_num in range(number_of_classes):
            if class_num not in label_value_counts.keys():
                missing_classes.append(class_num)

        if (len(missing_classes) > 0):
            logger.error(f"{set_type.capitalize()} set doesn't contain the following classes: {missing_classes}")

    def __str__(self):
        '''String representation of this dataset'''
        dataset_type = "Train" if self.train else "Val"
        output = {"size": len(self.data),
                  "label_distribution": {json.dumps(self.label_value_counts)}}
        return (f"{dataset_type}: {output}")