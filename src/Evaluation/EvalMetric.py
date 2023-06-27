from abc import ABC, abstractmethod
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Union
from functools import lru_cache

import eval4ner.muc as muc
import numpy as np
from src.Evaluation.EvalOutput import EvalOutput
from src.Evaluation import TokensToSentence


class EvalMetric(ABC):
    '''
    Calculate metric given the true and predicted labels.
    '''
    @abstractmethod
    def compute(self,
                eval_output: EvalOutput) -> Dict[str, Dict]:
        '''
        Standard classification metrics returned in a dictionary format.
        '''
        return NotImplementedError


class SentenceLevelMetric(EvalMetric):
    '''
    Calculate all the basic classification metrics on the sentence level based on some aggregation strategy on the predicted tokens.
    '''

    def __init__(self,
                 token_to_sentence_strategies: List[str],
                 token_to_sentence_strategies_init: List[dict]):
        '''
        Initializes the sentence level classification metrics.

        Parameter:
        ----------
        token_to_sentence_strategies: List[str]
            Aggregate token level labels to sentence level.
        token_to_sentence_strategies_init: List[dict]
            Init args for the strategies
        '''
        aggregation_methods = []
        for token_to_sentence_strategy, args in zip(token_to_sentence_strategies, token_to_sentence_strategies_init):
            try:
                aggregation_method = getattr(
                    TokensToSentence, token_to_sentence_strategy)
                aggregation_methods.append(aggregation_method(**args))
            except AttributeError:
                logger.error(
                    f"Token to sentence strategy '{token_to_sentence_strategy}' does not exist!")

        self.token_to_sentence_aggregation_methods = aggregation_methods
        self.token_to_sentence_strategies = token_to_sentence_strategies

    def compute(self,
                eval_output: EvalOutput) -> Dict[str, Dict]:
        '''
        Standard classification metrics returned in a dictionary format.
        '''
        res = {}

        for name, token_to_sentence in zip(self.token_to_sentence_strategies, self.token_to_sentence_aggregation_methods):

            y_true = token_to_sentence(eval_output.true_labels)
            y_pred = token_to_sentence(eval_output.predicted_labels)
            val_metric = calculate(y_true, y_pred)

            res[name] = flatten_dict(val_metric)

        return res


def calculate(y_true: np.array, y_pred: np.array) -> dict:
    '''
    Calculates all the basic classification metrics given the true and predicted labels.

    Parameters:
    ------------
    y_true: np.array
        True Labels
    y_pred: np.array
        Predicted labels

    Returns:
    ---------
    A dictionary containing:
    - each class's precision,  f1-score, support, accuracy.
    - overall accuracy
    - macro and weighted avg for precision, f1-score and support.
    '''
    output = classification_report(y_true=y_true,
                                   y_pred=y_pred,
                                   output_dict=True)

    # Add accuracy per class
    matrix = confusion_matrix(y_true=y_true,
                              y_pred=y_pred)

    label_accuracies = np.nan_to_num(matrix.diagonal() / matrix.sum(axis=0))

    for i, label_accuracy in enumerate(label_accuracies):
        label = f"{i}"
        if (label in output):
            output[label]["accuracy"] = label_accuracy

    return output


def flatten_dict(metrics: Dict[str, Union[float, Dict[str, float]]]) -> Dict[str, float]:
    '''
    Flattens the resulting nested metrics dictionary.
    '''
    output = {}

    for key, value in metrics.items():
        if (type(value) == dict):
            for key_l2, value_l2 in value.items():
                output[f"{key}_{key_l2}"] = value_l2
        else:
            output[key] = value
    return output


class TokenLevelMetric(EvalMetric):
    '''
    Calculates the metric on a token level.

    This includes calculating all normal metrics for each category.
    We also calculate the SemEval metric for `strict`, `exact`, `partial` and `type`.
    For the second one, we would also need the text.
    '''

    def __init__(self, tagging_system: str = "None"):
        '''
        Initializes the token level metric

        Parameter:
        -----------
        tagging_system: str
            The tagging system used on the token. Can be of the following: ['BIO', 'BILOU', 'None']

        '''
        self.tagging_system = tagging_system

    def compute(self,
                eval_output: EvalOutput) -> Dict[str, Dict]:
        '''
        Computes the token level metric.
        '''

        # First calculate the standard metrics on all categories (regardless of tagging system)
        flatten_true_labels = []
        flatten_predicted_labels = []

        ground_truths = []
        predictions = []
        texts = []

        for true_label, predicted_label in zip(eval_output.true_labels, eval_output.predicted_labels):
            flatten_true_labels += list(true_label)
            flatten_predicted_labels += list(predicted_label)

            true_label = self.convert_tags(true_label)
            predicted_label = self.convert_tags(predicted_label)

            ground_truths.append(self.collapse_labels(true_label))
            predictions.append(self.collapse_labels(predicted_label))
            texts.append(self.get_full_text(len(true_label)))

        val_metric = calculate(flatten_true_labels, flatten_predicted_labels)
        res = flatten_dict(val_metric)

        # Calculate exact, partial, type, strict
        semEvalMetrics = muc.evaluate_all(predictions, ground_truths, texts)
        semEvalMetrics = flatten_dict(semEvalMetrics)

        return {"TokenLevelMetric": {**res, **semEvalMetrics}}

    @staticmethod
    def collapse_labels(labels: List[int],
                        first_toxic_label: int = 1) -> List[Tuple[int, str]]:
        '''
        Collapses labels based on the following:
            Toxic_label, "text"

        Parameter:
        ---------
        labels: List[int]
            The labels to be converted
        first_toxic_label: int
            The first toxic label. Defaults to 1.

        Example:
        >>> labels = [0, 1, 2, 1, 1]
        >>> collapse_labels([0, 1, 2, 1, 1])
        [(1, ' b'), (2, ' c'), (1, ' d e')]
        '''

        if (len(labels) == 0):
            return []

        result = []
        prev_label = labels[0]
        text = f"{chr(97)}"

        for i in range(1, len(labels)):

            label = labels[i]
            if label != prev_label:
                if (prev_label >= first_toxic_label):
                    result.append((prev_label, text))

                prev_label = label
                text = ""

            text += f" {chr(i + 97)}"

        if (prev_label >= first_toxic_label):
            result.append((prev_label, text))
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def get_full_text(full_text_length: int):
        '''
        Retrieves the full text based on the length.
        '''
        if (full_text_length == 0):
            return ""

        return f"{TokenLevelMetric.get_full_text(full_text_length - 1)} {chr(full_text_length)}"

    def convert_tags(self, labels):
        '''
        Converts tags based on tagging system.

        ['BIO', 'BILOU', 'None']

        If tagging system is None, do nothing.

        '''
        if (self.tagging_system == "None"):
            return labels

        # If tagging system is BIO:
            # 0 -> 0
            # 1 -> 1
            # 2 -> 1
            # 3 -> 2
            # 4 -> 2
            # 5 -> 3
            # 6 -> 3
        if (self.tagging_system == "BIO"):
            return [((label + 1) // 2) for label in labels]

        # If tagging system is BILOU:
            # 0 -> 0
            # 1 -> 1
            # 4 -> 1
            # 5 -> 2
            # 8 -> 2
            # 9 -> 3
            # 12 -> 3
        if (self.tagging_system == "BILOU"):
            return [((label + 3) // 4) for label in labels]
