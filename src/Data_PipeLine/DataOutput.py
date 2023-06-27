from abc import ABC, abstractmethod
from random import random
from typing import List
from sklearn.model_selection import train_test_split
from loguru import logger

import pandas as pd
import numpy as np


class DataOutput(ABC):
    """
    Data Output
    """

    def __init__(self):
        return

    @abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        Takes the df and outputs it.

        Parameter:
        ----------
        df: pd.DataFrame
            Input DataFrame
        """
        raise NotImplementedError


class SplitDataset(DataOutput):
    """
    Retrieve the data from the BIG db.
    """

    def __init__(
        self,
        train_percentage: float = 0.8,
        test_percentage: float = 0.2,
        random_state: int = None,
        shuffle: bool = True,
        conversation_grouper: List[str] = None,
        ratio_tolerance: float = 0.05,
    ):
        """
        Initializes splitting of the dataset.
        Wrapper of sklearn.model_selection.train_test_split

        Parameter:
        ---------
        train_percentage: float
            Percentage of the training set
        test_percentage: float
            Percentage of the val set
        random_state: int
            Defaults to None. Controls the shuffling to be reproducible
        shuffle: bool
            Defaults to True. Whether to shuffle or not.
        conversation_grouper: List[str]
            Defaults to None. Defines what is a conversation.
        ratio_tolerance: float
            Defaults to 0.1. We check the following:
                1) (len(train) / len(train) + len(test)) ~ train_percentage +- ratio_tolerance
                2) for each category:
                    (len(train_category) / len(train_category) + len(test_category)) ~ train_percentage +- ratio_tolerance

            If tolerance is not met, we redo and increase random state by 1.
        """
        super().__init__()

        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.random_state = random_state
        self.shuffle = shuffle
        self.conversation_grouper = conversation_grouper
        if self.conversation_grouper is None:
            self.conversation_grouper = ["matchid"]
        self.ratio_tolerance = ratio_tolerance

    def __str__(self):
        return "SplitDataset"

    def __call__(self, df: pd.DataFrame, random_state: int = None):
        """
        Splits the DataFrame into train and validation set
        """
        df = df.copy()

        matchids = []
        avg_category_ids = []

        for i, group in df.groupby(self.conversation_grouper):
            matchids.append(i)

            toxic_categories = np.where(
                group["min_category_id"] != 0, group["min_category_id"], np.nan
            )
            avg_category_id = np.nanmean(toxic_categories)
            avg_category_id = 0 if np.isnan(avg_category_id) else int(avg_category_id)
            avg_category_ids.append(avg_category_id)

        if random_state is None:
            random_state = self.random_state
        logger.info(f"Splitting with random seed of: {random_state}")

        splits = train_test_split(
            matchids,
            train_size=self.train_percentage,
            test_size=self.test_percentage,
            random_state=random_state,
            shuffle=self.shuffle,
            stratify=avg_category_ids,
        )

        train = df[df["matchid"].isin(splits[0])]
        test = df[df["matchid"].isin(splits[1])]

        # Check #1: train and test length is split properly.
        train_length = len(train)
        test_length = len(test)

        length_ratio = train_length / (train_length + test_length)
        logger.info(f"Train-test ratio: {length_ratio}")
        if (
            length_ratio < self.train_percentage - self.ratio_tolerance
            or length_ratio > self.train_percentage + self.ratio_tolerance
        ):
            return SplitDataset(
                self.train_percentage,
                self.test_percentage,
                self.random_state + 1,
                self.shuffle,
                self.conversation_grouper,
                self.ratio_tolerance,
            )(df)

        # Check each category
        for i in test["min_category_id"].unique():
            train_toxic_category = (train["min_category_id"] == i).sum()
            test_toxic_category = (test["min_category_id"] == i).sum()
            toxic_category_ratio = train_toxic_category / (
                train_toxic_category + test_toxic_category
            )

            logger.info(f"Train-test category {i} ratio: {toxic_category_ratio}")

            if (
                toxic_category_ratio < self.train_percentage - self.ratio_tolerance
                or toxic_category_ratio > self.train_percentage + self.ratio_tolerance
            ):
                return SplitDataset(
                    self.train_percentage,
                    self.test_percentage,
                    self.random_state + 1,
                    self.shuffle,
                    self.conversation_grouper,
                    self.ratio_tolerance,
                )(df)

        return train, test
