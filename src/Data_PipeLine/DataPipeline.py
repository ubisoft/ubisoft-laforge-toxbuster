from loguru import logger
from typing import List, Tuple

import json
import pandas as pd
import inspect

from src.Data_PipeLine import DataRetrieval
from src.Data_PipeLine import DataTransformation
from src.Data_PipeLine import DataOutput


class DataPipeline():
    '''
    Builds the overall data pipeline. A data pipeline consists of
    1. One data retrieval method.
    2. A list of potential data transformation methods.
    3. One data output method.
    '''
    def __init__(self, data_retrieval_method: str,
                       data_retrieval_init_args: dict,
                       data_transformations: List[Tuple[str, dict]],
                       data_output_method: str = None,
                       data_output_init_args: dict =  None,
                ):
        '''
        Builds the data pipeline appropriately.
        '''

        try:
            data_retrieval = getattr(DataRetrieval, data_retrieval_method)
            self.data_retrieval = data_retrieval(**data_retrieval_init_args)
        except AttributeError:
            error_message = f"Data Retrieval method '{data_retrieval_method}' does not exist!"
            logger.error(error_message)
            raise ValueError(error_message)

        self.data_transformations = []

        for data_transformation_method, data_transformation_init_args in data_transformations:
            try:
                data_transformation = getattr(DataTransformation, data_transformation_method)
                self.data_transformations.append(data_transformation(**data_transformation_init_args))
            except AttributeError:
                error_message = f"Data Transformation method '{data_transformation_method}' does not exist!"
                logger.error(error_message)
                raise ValueError(error_message)

        self.data_output = None
        if (data_output_method is not None and data_output_method != ""):
            try:
                data_output = getattr(DataOutput, data_output_method)
                self.data_output = data_output(**data_output_init_args)
            except AttributeError:
                error_message = f"Data Output method: `{data_output_method}` does not exist!"
                logger.error(error_message)
                raise ValueError(error_message)

    def __call__(self) -> pd.DataFrame:
        '''
        Runs the data pipeline in the following order:

            Data Retrieval -> Data Transformation 1 -> Data Transformation 2 -> ...

        Underneath, we are passing a pd.DataFrame from one step to the other.
        '''
        output = self.data_retrieval()
        logger.debug(f"Retrieved data of size {output.shape}.")

        for data_transformation in self.data_transformations:
            output = data_transformation(output)
            logger.debug(f"After {data_transformation}, data is of size: {output.shape}")

        if (self.data_output is not None):
            output = self.data_output(output)
            logger.debug(f"After {self.data_output}, we get {type(output)}.")

        return output

    @staticmethod
    def from_config_file(file_name: str):
        '''
        Initialise the DataPipeline from a config file.

        Parameter:
        ---------
        file_name: str
            Path to the data_pipeline config file
        '''
        with open(file_name) as f:
            config = json.load(f)
        return DataPipeline.from_config(config)

    @staticmethod
    def from_config(config: dict):
        '''
        Initialise the DataPipeline from a config.

        Parameter:
        ---------
        config: dict
            Path to the data_pipeline config file
        '''
        init_arguments = inspect.getfullargspec(DataPipeline.__init__)
        needed_args = init_arguments.args[1:] # skip the self

        filtered_config = {}
        for needed_arg in needed_args:
            filtered_config[needed_arg] = config[needed_arg]

        return DataPipeline(**filtered_config)