import torch
from loguru import logger
from transformers import (AutoConfig, AutoModelForSequenceClassification)


class ToxicSentenceClassification(torch.nn.Module):
    '''
    Wrapper class for AutoModelForSequenceClassification. No custom code for this yet.

    Attributes:
    -----------
    model_name: str
        The name of the pre-trained model.
    num_labels: int
        The number of labels.
    config: AutoConfig
        The loaded configs based on the model_name
    lm: AutoModelForTokenClassification
        HuggingFace default model for Token Classification.
    '''

    def __init__(self, model_name: str, num_labels: int, num_additional_tokens: int = 0):
        '''
        Initializes the model

        Parameter:
        ----------
        model_name: str
            Name of the pre-trained model / path to the pre-trained_model.
        num_labels: int
            Number of labels to classify.
        '''

        super(ToxicSentenceClassification, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # A bug where you can't pass num_labels if call
        # AutoModelForTokenClassification.from_config(self.config)
        self.config = AutoConfig.from_pretrained(model_name)
        self.lm = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)

        if (num_additional_tokens > 0):
            new_vocab_size = self.config.vocab_size + num_additional_tokens
            self.lm.resize_token_embeddings(new_vocab_size)

            logger.info(
                f"Updated model vocab size to be: {new_vocab_size} (+{num_additional_tokens})")

    def forward(self, **args):
        return self.lm(**args)
