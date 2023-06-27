import json
import os
from pathlib import Path
from loguru import logger

import pandas as pd

from src.Data.Dataset import ToxDataset
from src.Data.utils import collate_init
from src.Data_PipeLine.DataPipeline import DataPipeline
from src.Evaluation import Evaluator
from src.Evaluation.EvalOutput import update_and_save_to_csv
from src.Benchmark.BenchmarkerConfig import BenchmarkerConfig
from src.Models.utils import load_checkpoint, model_init


class Benchmarker:
    '''
    Main Benchmark driver.
    '''

    def __init__(self, config: BenchmarkerConfig):
        '''
        Initializes the Benchmarker in the following order:

        1. Save Config file
        2. Process Data Pipeline
        3. Initialize collate function
        4. Initialize model & load checkpoint
        5. Initialize validation pipeline

        Parameter:
        ----------
        config: BenchmarkerConfig
            Config to be used by the Benchmarker.
        '''
        self.config = config
        benchmark_file_name = os.path.splitext(config.save_file_name)[0]
        self.config.save(
            f"{config.save_folder_path}/{benchmark_file_name}_benchmarker_config.json")

        # Process Data Pipeline -> DataSet, Instantiate collate
        process_data = DataPipeline.from_config({**config._asdict()})

        data_output = process_data()

        if (type(data_output) == pd.DataFrame):
            self.df = data_output
        elif (type(data_output) == tuple):
            self.df = data_output[1]

        self.dataset = ToxDataset(self.df)

        logger.info(self.dataset)
        # Override the train to be False.
        # Useful if we are passing the config file used in Training :D
        self.collate = collate_init(collate_method_name=config.collate_method_name,
                                    collate_method_init_args={**config.collate_method_init_args})

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

        self.eval_metric_pipeline = Evaluator.EvalMetricPipeline(
            config.val_metric_methods)

    def benchmark(self) -> pd.DataFrame:
        '''
        Main driver to infer based on the provided model, dataset, collate function.
        Adds columns `predictions` and `confidence_levels` to the dataset and saves it into `;`-separated file.

        '''
        output = Evaluator.predict_labels(model=self.model,
                                          dataset=self.dataset,
                                          collate=self.collate,
                                          batch_size=self.config.val_batch_size,
                                          num_workers=self.config.val_dataloader_num_workers)

        eval_metrics = self.eval_metric_pipeline.run(output,
                                                     tb_writer=None,
                                                     output_type="Benchmark",
                                                     epoch = -1)

        file_name = f"{self.config.save_folder_path}/{self.config.metric_result_save_file_name}"
        Path(self.config.save_folder_path).mkdir(parents=True, exist_ok=True)
        self.eval_metric_pipeline.save(eval_metrics,
                                       metric_result_save_file_name = file_name)

        df = update_and_save_to_csv(output = output,
                                    df = self.df,
                                    save_folder_name=self.config.save_folder_path,
                                    save_file_name=self.config.predicted_labels_save_file_name,
                                    include_confidence_levels=self.config.confidence_levels,
                                    include_per_class_confidence_levels=self.config.per_class_confidence_levels,
                                    include_true_labels=True,
                                    )

        return eval_metrics, df
