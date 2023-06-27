import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.Data.Dataset import ToxDataset
from src.Data.utils import collate_init, get_num_of_classes
from src.Data_PipeLine.DataOutput import SplitDataset
from src.Data_PipeLine.DataPipeline import DataPipeline
from src.Evaluation import Evaluator
from src.Models.utils import load_checkpoint, model_init, save_checkpoint
from src.Training import EarlyStop, utils
from src.Training.EarlyStop import EarlyStoppingStrategy, NoEarlyStopping
from src.Training.TrainerConfig import TrainerConfig

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch import nn


class Trainer:
    """
    Main training driver that takes in any torch module.

    Attributes:
    -----------
    model: torch.nn.Module
        Torch model that will be trained.
    config: TrainerConfig
        All the configs related to training.
    collate: Collate
        The collate function passed to DataLoader - used to convert raw input to tokens & correct label per token.
    n_gpus: int
        Number of gpus for this trainer
    device: torch.Device
        The device(s) we will be training this model.
    optimizer: torch.optim.Optimizer
        AdamW optimizer for the model
    batch_size: int
        Number of samples in a batch.
    models_saved: list[str]
        Path to all the models saved/
    train_dataset: ToxDataset
        Training Dataset
    val_dataset: ToxDataset
        Validation Dataset
    best_model_name: str
        File name to the best model name
    model_name: str
        File name of the model
    starting_epoch: int
        Starting epoch. Defaults to 0.
    best_model_metric:
        The metric value for the best model. Defaults to -math.inf
    lr_scheduler: LambdaR
        Scheduler for the learning rate.
    models_saved: List[str]
        File names of models saved
    val_outputs_saved: List[str]
        File names of validation outputs saved.
    earlyStop: EarlyStoppingStrategy
        Strategy for early stopping
    eval_metric_pipeline: EvalMetricPipeline
        Pipeline for evaluating the metrics.

    """

    def __init__(self, config: TrainerConfig):
        """
        Initializes the Trainer using the provided config in the following order.
        * Load Computer Specs
        * Process DataPipeline to get appropriate datasets

        * Initialize Model, Optimizer, LR Scheduler
        * Initialize Validator
        * Initialize Early Stop Strategies

        Parameter:
        ------------
        config: TrainerConfig
            Configurations for this Trainer
        """
        self.config = config
        self.best_model_name = self.__get_best_model_name()
        self.model_name = self.__get_model_name()
        self.config.save(f"{self.best_model_name}_trainer_config.json")

        # Load computer specs
        self.n_gpus, self.device = utils.gpu_information_summary()
        utils.set_random_seed(self.config.random_seed, self.n_gpus)
        self.batch_size = config.per_gpu_train_batch_size * max(1, self.n_gpus)

        # Process Data Pipeline -> DataSet, Instantiate collate
        process_data = DataPipeline.from_config(config._asdict())
        train_df, test_df = process_data()
        train_df, val_df = SplitDataset(
            train_percentage=(1 - self.config.val_percentage),
            test_percentage=self.config.val_percentage,
            random_state=self.config.random_seed,
            shuffle=True,
            ratio_tolerance=self.config.val_ratio_tolerance,
        )(train_df)

        self.train_dataset = ToxDataset(train_df)
        self.val_dataset = ToxDataset(val_df)
        self.test_dataset = ToxDataset(test_df)
        logger.info(self.train_dataset)
        logger.info(self.val_dataset)
        logger.info(self.test_dataset)
        self.num_classes = get_num_of_classes(self.train_dataset, self.val_dataset)

        logger.debug(config.collate_method_init_args)
        self.collate = collate_init(
            collate_method_name=config.collate_method_name,
            collate_method_init_args=config.collate_method_init_args,
        )

        # Initialize Model, Optimizer, LR Scheduler, Continuous Training
        self.model = model_init(
            model_classification_type=config.model_classification_type,
            model_class=config.model_class,
            model_class_init_args={
                "num_labels": self.num_classes,
                **config.model_class_init_args,
            },
        )
        self.model = self.model.to(self.device)

        self.optimizer = self.__setup_optimizer()
        self.lr_scheduler = self.__setup_lr_scheduler()
        self.starting_epoch, self.best_model_metric = self.__setup_continue_learning()

        # Move loading of checkpoint before we go to DataParallel
        if self.n_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Instantiate Validator
        self.eval_metric_pipeline = Evaluator.EvalMetricPipeline(
            config.val_metric_methods
        )
        self.earlyStop = self.__setup_early_stop()

        self.models_saved = []
        self.val_outputs_saved = []

    def __setup_optimizer(self):
        """
        Setup the AdamW Optimizer based on the model existing parameters add weight decay
        """
        # These parameters have no decay in the original implementation
        # so we group them the same way
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

    def __setup_lr_scheduler(self):
        """
        Sets up the learning rate scheduler to be used based on the configs.

        Returns:
        --------
        A learning rate scheduler.
        If no hard restart cycles, we use a linear schedule with warmups.
        If there are hard restart cycles, we use cosine schedule with hard restarts.

        """

        num_training_step_per_epoch = len(self.train_dataset) // self.batch_size
        total_training_steps = (
            num_training_step_per_epoch * self.config.max_epochs_to_train
        )
        total_global_steps = (
            total_training_steps // self.config.gradient_accumulation_steps
        )

        # Linear schedule if no hard restart cycles.
        if self.config.num_hard_restart_cycles <= 0:
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=total_global_steps,
            )

        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=total_global_steps,
            num_cycles=self.config.num_hard_restart_cycles,
        )

    def __setup_early_stop(self) -> EarlyStoppingStrategy:
        """
        Initializes early stop strategy.
        """
        early_stop = NoEarlyStopping()

        if self.config.early_stopping_strategy == "":
            return early_stop

        try:
            early_stopping_strategy = getattr(
                EarlyStop, self.config.early_stopping_strategy
            )
            early_stop = early_stopping_strategy(
                min_epochs_to_train=self.config.min_epochs_to_train,
                metric_to_compare=self.config.metric_to_compare,
                patience=self.config.patience,
            )
        except AttributeError:
            logger.error(
                f"Early Stop Strategy '{self.config.early_stopping_strategy}' does not exist!"
            )

        return early_stop

    def __setup_continue_learning(self) -> Tuple[int, float]:
        """
        Setups the model if we want to continue learning.

        Depends on the following configs:
        1. `continue_training`
        2. `continue_training_model_file_path`.

        Returns the starting epoch and loads the states to the model & optimizer if we are going to continue to train.
        """
        if not self.config.continue_training:
            return 0, -math.inf

        model_name = self.config.continue_training_model_file_path
        if model_name == "":
            model_name = f"{self.best_model_name}.pt"

        if not os.path.exists(model_name):
            logger.error(f"Could not find checkpoint `{model_name}` to load")
            return 0, -math.inf

        self.model, self.optimizer, epoch, best_model_metric = load_checkpoint(
            model=self.model,
            checkpoint_path=model_name,
            optimizer=self.optimizer,
            best_model_metric_name=self.config.save_best_model_metric,
        )
        epoch = max(0, epoch)
        logger.info(f"Continue training with loaded checkpoint {self.model_name}.pt")

        return epoch, best_model_metric

    def __get_best_model_name(self) -> str:
        """Gets the best model name"""

        model_name = self.config.best_model_save_name

        if model_name == "":
            model_name = self.config.model_class

        str_date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

        return f"{self.config.save_folder_path}/{str_date}/{model_name}"

    def __get_model_name(self) -> str:
        model_name = (
            self.config.model_class
            if self.config.model_save_prepend_name == ""
            else self.config.model_save_prepend_name
        )
        str_date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        return f"{self.config.save_folder_path}/{str_date}/{model_name}"

    def train(self):
        """
        Trains the model for `num_train_epochs`.
        Gradient accumulation is used for distributed GPU training and increasing the effective batch size of the model.
        Gradient is accumulated for `gradient_accumulation_steps`.
        We also clip and normalize the gradient to prevent vanishing / exploding gradient.


        Validation loop on model will be run based on:
        - `evaluate_during_training`
        - `prediction_evaluator` is set.
        - `num_epochs_to_eval`

        Save model based on:
        - `save_model`
        - `num_epoch_to_save`
        - `num_models_to_save`
        - `save_folder_path`
        """
        tb_writer = SummaryWriter(log_dir=self.config.tb_log_dir)

        # Setup train
        train_batch_size = self.config.per_gpu_train_batch_size * max(1, self.n_gpus)

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            sampler=RandomSampler(self.train_dataset),
            num_workers=self.config.train_dataloader_num_workers,
            collate_fn=self.collate,
        )

        if self.config.weighted_loss:
            all_labels_flattened = []
            for instance_labels in self.train_dataset.data["label"]:
                for label in instance_labels:
                    all_labels_flattened.append(label)
            all_labels_flattened.append(6)

            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=[i for i in range(self.num_classes)],
                y=all_labels_flattened,
            )

        self.__log_train_start()

        # Initialize Variables
        self.model.zero_grad()
        global_step_loss = 0.0
        global_step = 1

        train_iterator = trange(
            self.starting_epoch, self.config.max_epochs_to_train, desc="Epoch"
        )
        for epoch_num in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc=f"Iteration {epoch_num}", disable=False
            )
            for step, batch in enumerate(epoch_iterator):
                self.model.train()

                input = {}
                for key, value in batch.items():
                    if key != "prediction_mask":
                        input[key] = value.to(self.device)

                outputs = self.model(**input)
                loss = outputs[0]

                if self.config.weighted_loss:
                    logits = outputs.get("logits")
                    labels = input.get("labels")
                    loss_function = nn.CrossEntropyLoss(
                        weight=torch.FloatTensor(class_weights).to(self.device)
                    )
                    loss = loss_function(
                        logits.view(-1, self.num_classes), labels.view(-1)
                    )

                # Calculate Loss
                if self.n_gpus > 1:
                    loss = loss.mean()  # Average over multi-gpu
                # Normalize loss over the batch size.
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                global_step_loss += loss.item()

                # Gradient Accumulation.
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip to prevent vanishing and exploding gradients.
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()

                    tb_writer.add_scalar(
                        tag="Loss/Train",
                        scalar_value=global_step_loss
                        / self.config.gradient_accumulation_steps,
                        global_step=global_step,
                    )
                    tb_writer.add_scalar(
                        tag="Learning Rate",
                        scalar_value=self.optimizer.param_groups[0]["lr"],
                        global_step=global_step,
                    )

                    global_step_loss = 0.0
                    global_step += 1

            # Logging step
            if epoch_num % self.config.log_per_x_epochs == 0:
                logger.debug(f"Epoch {epoch_num} has finished running.")

            overall_val_metrics = self.__val_loop(tb_writer=tb_writer, epoch=epoch_num)
            self.__save_model(epoch=epoch_num, overall_val_metric=overall_val_metrics)
            self.__save_best_model(
                epoch=epoch_num, overall_val_metric=overall_val_metrics
            )

            if self.__check_early_stop(
                epoch=epoch_num, overall_val_metric=overall_val_metrics
            ):
                logger.info(f"Satisfied early stop at epoch: {epoch_num}")
                break

        self.__test_loop()

        tb_writer.close()
        self.__log_train_end()

    def __log_train_start(self):
        """
        Logs basic info when training starts.
        """
        logger.info("***** Running training *****")
        logger.info(f"\tTraining Size: {len(self.train_dataset)}")
        logger.info(f"\tStarting Epochs: {self.starting_epoch}")
        logger.info(f"\tMax Number of Epochs: {self.config.max_epochs_to_train}")
        logger.info(f"\tBatch Size Per GPU: {self.config.per_gpu_train_batch_size}")
        logger.info(
            f"\tGradient Accumulation Steps = {self.config.gradient_accumulation_steps}"
        )
        logger.info(
            f"\tBest Model `{self.config.save_best_model_metric}` Initial Value: {self.best_model_metric}"
        )

        if self.val_dataset:
            logger.info(f"\tValidation Size: {len(self.val_dataset)}")
            logger.info(f"\tEarly Stopping Strategy: {self.earlyStop}")

    def __val_loop(self, tb_writer: SummaryWriter, epoch: int) -> Dict[str, float]:
        """
        Run model against validation dataset and record to TensorBoard if the following conditions are met:
        1. Config `val` is set to True.
        2. In DataPipeline, the val_dataset is not None
        3. Evaluates every `val_per_x_epochs`.

        Parameters:
        ----------
        tb_writer: SummaryWriter
            Write to logs for TensorBoard.
        epoch: int
            Epoch
        """
        if (
            not self.config.val
            or not self.val_dataset
            or (epoch % self.config.val_per_x_epochs != 0)
        ):
            return {}

        val_output = Evaluator.predict_labels(
            model=self.model,
            dataset=self.val_dataset,
            collate=self.collate,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.val_dataloader_num_workers,
        )

        tb_writer.add_scalar(
            tag="Loss/Val", scalar_value=val_output.loss, global_step=epoch
        )

        eval_metrics = self.eval_metric_pipeline.run(
            eval_output=val_output, tb_writer=tb_writer, output_type="Val", epoch=epoch
        )
        return eval_metrics

    def __check_early_stop(
        self, epoch: int, overall_val_metric: Dict[str, float]
    ) -> bool:
        """'
        Check if early stop or not.

        Parameter:
        ---------
        Checks if we need to early stop or not.
        """
        if len(overall_val_metric) == 0:
            return False

        return self.earlyStop(epoch=epoch, val_metrics=overall_val_metric)

    def __save_model(self, epoch: int, overall_val_metric: Dict[str, float]):
        """
        Saves the model if it passes all the requirements.
        1. Config `save_model` is set to `True`
        2. Current step is a multiple of `saving_step`.
        3. Remove earlier models if we have saved more than `num_models_to_save`.

        Model save name is determined by `model_save_prepend_name`.
        If that is "", we use `model_name`.

        Parameter:
        ----------
        epoch: int
            Current step of the training.
        overall_val_metric: Dict[str, float]
            Overall validation metric to be saved.
        """
        if (
            not self.config.save_model
            or epoch % self.config.save_per_x_epoch != 0
            or overall_val_metric == {}
        ):
            return

        model_checkpoint_name = f"{self.model_name}_{epoch}.pt"
        val_metric_save_file = f"{self.model_name}_{epoch}_val_result.json"

        best_model_metric = (
            overall_val_metric[self.config.save_best_model_metric]
            if self.config.save_best_model_metric in overall_val_metric
            else -100
        )

        save_checkpoint(
            epoch=epoch,
            model=self.model,
            model_checkpoint_name=model_checkpoint_name,
            optimizer=self.optimizer,
            best_model_metric_name=self.config.save_best_model_metric,
            best_model_metric=best_model_metric,
            save_folder_path=self.config.save_folder_path,
        )

        with open(val_metric_save_file, "w") as json_file:
            json.dump(overall_val_metric, json_file, indent=1)

        self.models_saved.append(model_checkpoint_name)
        self.val_outputs_saved.append(val_metric_save_file)

        # Remove models saved in FIFO order if we have saved more models than set in configuration.
        if len(self.models_saved) > self.config.num_models_to_save:
            model_to_be_removed = self.models_saved.pop(0)
            os.remove(model_to_be_removed)

            val_output_to_be_removed = self.val_outputs_saved.pop(0)
            os.remove(val_output_to_be_removed)

    def __save_best_model(self, epoch: int, overall_val_metric: Dict[str, float]):
        """
        Save the best model seen so far based on `save_best_model_metric` in the config file.
        Model name is currently saved using `best_model_save_name`. If that is "", then we use `model_name`.
        The output is saved as the "{model_name}_result.json".

        Depends on the following:
        -----------------------------------
        `save_best_model_metric`: str
            If empty / not in validation metrics, will not save.
        `min_epochs_to_train`: int
            Will not save model before the minimum epochs to train.

        Parameter:
        ----------
        epoch: int
            Current epoch number.
        overall_val_metric: Dict[str, float]
            Overall validation metric. An empty dict means this epoch wasn't validated.
        """
        if (
            self.config.save_best_model_metric == ""
            or overall_val_metric == {}
            or self.config.save_best_model_metric not in overall_val_metric
        ):
            return

        new_model_metric = overall_val_metric[self.config.save_best_model_metric]
        if self.best_model_metric < new_model_metric:
            self.best_model_metric = new_model_metric

            model_checkpoint_name = f"{self.best_model_name}.pt"
            val_metric_save_file = f"{self.best_model_name}_val_result.json"

            save_checkpoint(
                epoch=epoch,
                model=self.model,
                model_checkpoint_name=model_checkpoint_name,
                optimizer=self.optimizer,
                best_model_metric_name=self.config.save_best_model_metric,
                best_model_metric=new_model_metric,
                save_folder_path=self.config.save_folder_path,
            )

            with open(val_metric_save_file, "w") as json_file:
                json.dump(overall_val_metric, json_file, indent=1)

            logger.info(
                f"Saved best model at epoch `{epoch}` with new value of `{new_model_metric}` for {self.config.save_best_model_metric}."
            )

    def __test_loop(self):
        test_output = Evaluator.predict_labels(
            model=self.model,
            dataset=self.test_dataset,
            collate=self.collate,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.val_dataloader_num_workers,
        )

        eval_metrics = self.eval_metric_pipeline.run(
            eval_output=test_output, tb_writer=None, output_type="Test", epoch=-1
        )

        test_metric_save_file = f"{self.best_model_name}_test_result.json"

        self.eval_metric_pipeline.save(
            eval_metrics, metric_result_save_file_name=test_metric_save_file
        )

    def __log_train_end(self):
        """
        Logs training end.
        """
        logger.info("Training is over.")
        logger.info("******************")
