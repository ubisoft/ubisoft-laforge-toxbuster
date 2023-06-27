import json
from loguru import logger
from torch.utils.data import Dataset
from torch.nn.functional import softmax
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.Data.Collate import Collate
from src.Evaluation import EvalMetric
from src.Evaluation.EvalOutput import EvalOutput
from src.Training.utils import gpu_information_summary, to_numpy



class EvalMetricPipeline:
    '''
    Evaluates the dataset from prediction to metrics.
    '''
    def __init__(self,
                 val_metric_methods: List[Tuple[str, dict]]
                 ):
        '''
        Initializes the pipeline to validate the metrics.
        '''

        self.val_methods: List[EvalMetric.EvalMetric] = []

        for (val_metric_method, val_metric_init_args) in val_metric_methods:
            try:
                val_method = getattr(EvalMetric, val_metric_method)
                self.val_methods.append(val_method(**val_metric_init_args))
            except AttributeError:
                logger.error(f"Validation strategy '{val_metric_method}' does not exist!")

    def run(self,
            eval_output: EvalOutput,
            tb_writer = None,
            output_type: str = "Val",
            epoch: int = 0,
            )-> List[Dict[str, dict]]:
        '''
        Runs the whole eval metric pipeline.

        Parameter:
        -----------
        eval_output: EvalOutput
            The output of the evaluation.
        '''

        flattened_overall_val_metrics = {}
        # Add loss to the output
        flattened_overall_val_metrics["loss"] = eval_output.loss

        # Loop through each evaluation method
        for val_method in self.val_methods:

            val_method_output = val_method.compute(eval_output)

            # For each evaluation method output, flatten to the valuation metric
            for name, val_metric in val_method_output.items():

                # if we passed in a tb_writer to tb_w
                if (tb_writer):
                    tb_writer.add_scalars(main_tag=f"{name}/{output_type}",
                                          tag_scalar_dict=val_metric,
                                          global_step=epoch)

                for metric in val_metric:
                    flattened_overall_val_metrics[f"{name}_{metric}"] = val_metric[metric]


        return flattened_overall_val_metrics

    def save(self, eval_metrics: dict, metric_result_save_file_name: str):
        with open(metric_result_save_file_name, mode="w") as f:
            json.dump(eval_metrics, f, indent=1)


def predict_labels(model: torch.nn.Module,
                   dataset: Dataset,
                   collate: Collate,
                   batch_size: int = 32,
                   num_workers: int = 0) -> EvalOutput:
    '''
    Predict the labels for the dataset.

    Parameter:
    -----------
    model: torch.nn.Module
        Model to use to predict the label
    dataset: torch.utils.data.Dataset
        The dataset.
    collate: Collate
        The function to collate the data
    batch_size: int
        Default to 32. Batch size to evaluate the dataset.
    num_workers: int
        Default to 0. Number of workers to process the dataset in the dataloader.

    Returns:
    --------
    EvalOutput with predicted_labels, true_labels and eval loss.
    Eval loss is meaningless if we set collate.is_train to false. (It returns a label mask).
    '''

    eval_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=collate)

    n_gpu, device = gpu_information_summary(show=False)
    model.to(device)
    model.eval()

    eval_loss = 0
    predictions = []
    predictions_confidence = []
    predictions_confidences = []
    true_labels = []

    with torch.inference_mode():
        for eval_instance in tqdm(eval_loader, desc=f"Eval"):

            batch = {}
            for key, value in eval_instance.items():
                if (key != "prediction_mask"):
                    batch[key] = value.to(device)
                else:
                    prediction_masks = value.to(device)

            outputs = model(**batch)
            loss = outputs[0]

            if n_gpu > 1:
                loss = loss.mean()
            eval_loss += loss.item()

            for labels, prediction_mask, logits in zip(batch["labels"], prediction_masks, outputs[1]):
                logits_clean = logits[prediction_mask == 1]

                sample_confidences = softmax(logits_clean, dim = 1)
                sample_confidence, sample_prediction = sample_confidences.max(dim=1)

                predictions.append(to_numpy(sample_prediction))
                predictions_confidence.append(to_numpy(sample_confidence))
                predictions_confidences.append(to_numpy(sample_confidences))
                true_labels.append(to_numpy(labels[prediction_mask == 1]))

        eval_loss = eval_loss / len(eval_loader)

        return EvalOutput(predicted_labels=predictions,
                          predicted_labels_confidence= predictions_confidence,
                          predicted_labels_confidences = predictions_confidences,
                          true_labels=true_labels,
                          loss = eval_loss,
                          )