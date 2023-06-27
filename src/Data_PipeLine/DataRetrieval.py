from abc import ABC, abstractmethod


import pandas as pd


class DataRetrieval(ABC):
    """
    Retrieval method of the data.
    """

    def __init__(self):
        return

    @abstractmethod
    def __call__(self) -> pd.DataFrame:
        """
        Retrieves the data and loads it into a pd.DataFrame
        """
        raise NotImplementedError


class RetrieveFromCSV(DataRetrieval):
    """
    Retrieves the data from an existing CSV file.
    """

    def __init__(self, file_name: str, separator: str = ";"):
        """
        Initializes the retrieval method.

        Parameter:
        -----------
        file_name: str
            Path to the file
        separator: str
            Defaults to ";". Separator of the CSV file.
        """
        super().__init__()
        self.file_name = file_name
        self.separator = separator

    def __call__(self) -> pd.DataFrame:
        """
        Loads the CSV file into a pd.Dataframe.
        """
        return pd.read_csv(self.file_name, sep=self.separator)
