from pathlib import Path
from loguru import logger

import os
import pandas as pd


from src.Data.Dataset import ToxDataset
from src.Data.utils import collate_init
from src.Data_PipeLine.DataPipeline import DataPipeline
from src.Evaluation import Evaluator
from src.Evaluation.EvalOutput import update_and_save_to_csv
from src.Infer.InfererConfig import InfererConfig
from src.Models.utils import load_checkpoint, model_init


class Inferer:
    '''
    Main Inference driver.
    '''

    def __init__(self, config: InfererConfig):
        '''
        Initializes the Inferer in the following order:

        1. Save Config file
        2. Process Data Pipeline
        3. Initialize collate function
        4. Initialize model & load checkpoint

        Parameter:
        ----------
        config: InfererConfig
            Config to be used by the Inferer.
        '''
        self.config = config
        infer_file_name = os.path.splitext(config.predicted_labels_save_file_name)[0]
        self.config.save(
            f"{config.save_folder_path}/{infer_file_name}_inferer_config.json")

        # Process Data Pipeline -> DataSet, Instantiate collate
        process_data = DataPipeline.from_config({**config._asdict()})

        data_output = process_data()

        if (type(data_output) == pd.DataFrame):
            self.df = data_output
        elif (type(data_output) == tuple):
            self.df = data_output[1]

        self.dataset = ToxDataset(self.df, train=False)

        logger.info(self.dataset)
        # Override the train to be False.
        # Useful if we are passing the config file used in Training :D
        self.collate = collate_init(collate_method_name=config.collate_method_name,
                                    collate_method_init_args={**config.collate_method_init_args, "train": False, })

        # Initialize Model
        self.model = model_init(
            model_classification_type=config.model_classification_type,
            model_class=config.model_class,
            model_class_init_args={
                "num_labels": config.num_classes, **config.model_class_init_args}
        )
        self.model, _, _, _ = load_checkpoint(model=self.model,
                                              checkpoint_path=config.checkpoint_path,
                                              optimizer=None,
                                              best_model_metric_name=None,
                                              )
        logger.info(f"Loaded checkpoint {config.checkpoint_path}")

    def infer(self) -> pd.DataFrame:
        '''
        Main driver to infer based on the provided model, dataset, collate function.
        Adds columns `predictions` and `confidence_levels` to the dataset and saves it into `;`-separated file.

        '''
        columns = list(self.df.columns)
        if self.config.confidence_levels:
            columns.append("Confidence Levels")
        if self.config.per_class_confidence_levels:
            columns.append("Confidence Level Per Class")
        logger.debug(f"Columns to save: {columns}")

        output = Evaluator.predict_labels(model=self.model,
                                          dataset=self.dataset,
                                          collate=self.collate,
                                          batch_size=self.config.val_batch_size,
                                          num_workers=self.config.val_dataloader_num_workers)

        df = update_and_save_to_csv(output = output,
                                    df = self.df,
                                    save_folder_name=self.config.save_folder_path,
                                    save_file_name=self.config.predicted_labels_save_file_name,
                                    include_confidence_levels=self.config.confidence_levels,
                                    include_per_class_confidence_levels=self.config.per_class_confidence_levels,
                                    include_true_labels=False,
                                    )
        return df
