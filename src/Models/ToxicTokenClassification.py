import torch
from loguru import logger
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          BertConfig, BertModel, BertPreTrainedModel)
from src.Models.BertWithSpeakerSegmentation import BertWithSpeakerSegmentationForTokenClassification
from src.Models.BertWithSpeakerSegmentationConfig import BertWithSpeakerSegmentationConfig

class ToxicTokenClassification(torch.nn.Module):
    '''
    Wrapper class for AutoModelForTokenClassification. No custom code for this yet.

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

        super(ToxicTokenClassification, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # A bug where you can't pass num_labels if call
        # AutoModelForTokenClassification.from_config(self.config)
        self.config = AutoConfig.from_pretrained(model_name)
        self.lm = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels)

        if (num_additional_tokens > 0):
            new_vocab_size = self.config.vocab_size + num_additional_tokens
            self.lm.resize_token_embeddings(new_vocab_size)

            logger.info(
                f"Updated model vocab size to be: {new_vocab_size} (+{num_additional_tokens})")

    def forward(self, **args):
        return self.lm(**args)


class BertWithSpeakerSegmentationToxicTokenClassification(torch.nn.Module):
    '''
    Wrapper class for BertWithSpeakerSegmentationForTokenClassification
    '''

    def __init__(self, num_labels: int):
        '''
        Initializes the model

        Parameter:
        ----------
        num_labels: int
            Number of labels to classify.
        '''
        super(BertWithSpeakerSegmentationToxicTokenClassification, self).__init__()

        self.model_name = "BertWithSpeakerSegmentation"
        self.num_labels = num_labels

        # A bug where you can't pass num_labels if call
        self.config = BertWithSpeakerSegmentationConfig(num_labels=num_labels)
        self.lm = BertWithSpeakerSegmentationForTokenClassification(self.config)

        # Load weights from bert-base-uncased
        logger.info("Trying to load bert-base-uncased weights")
        lm_temp = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        new_state_dict = {}
        state_dict = lm_temp.state_dict()
        for i in state_dict.keys():
            if i.startswith("bert"):
                new_state_dict[f"bert_with_speaker_segmentation{i[4:]}"] = state_dict[i]
            else:
                new_state_dict[i] = state_dict[i]

        self.lm.load_state_dict(new_state_dict, strict = False)

    def forward(self, **args):
        return self.lm(**args)


class BertToxicDetector(BertPreTrainedModel):
    '''
    Reza's model. Functionally the same as ToxicTokenClassification.
    '''

    def __init__(self, num_labels):
        config = BertConfig(num_labels=num_labels, classifier_dropout=0.5)
        super().__init__(config)
        self.num_labels = num_labels

        self.back_bone = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.task_classifier = torch.nn.Linear(config.hidden_size, num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.back_bone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.task_classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


