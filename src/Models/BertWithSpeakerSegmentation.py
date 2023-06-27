from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union
from torch import nn
from transformers.models.bert import modeling_bert
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutputWithPoolingAndCrossAttentions, TokenClassifierOutput)
from transformers.utils import ModelOutput
from torch.nn import CrossEntropyLoss


from src.Models.BertWithSpeakerSegmentationConfig import BertWithSpeakerSegmentationConfig

import torch


class BertWithSpeakerSegmentationEmbeddings(modeling_bert.BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):

        # Initiate BertEmbeddings config
        super().__init__(config)

        # Add our additional token embeddings
        self.player_ID_embeddings = nn.Embedding(
            config.player_ID_vocab_size, config.hidden_size)
        self.chat_type_embeddings = nn.Embedding(
            config.chat_type_vocab_size, config.hidden_size)
        self.team_type_embeddings = nn.Embedding(
            config.team_ID_vocab_size, config.hidden_size)

    def __set_default_if_none(self,
                              attribute_ids: Optional[torch.LongTensor],
                              attribute_name: str,
                              seq_length: int,
                              input_shape: Tuple[int])-> torch.LongTensor:
        '''
        Helper function to set the default attribute if provided is `None`.
        We use buffer already registered (token_type_id in modeling_bert.BertEmbeddings)
        '''
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664

        # Return original if already set
        if attribute_ids is not None:
            return attribute_ids

        # If we no attribute, we just set a set of zero.
        if not hasattr(self, attribute_name):
            return torch.zeros(input_shape,
                               dtype=torch.long,
                               device=self.position_ids.device)

        # Return expanded from buffer
        buffered_attribute_ids = self.token_type_ids[:, :seq_length]
        return buffered_attribute_ids.expand(input_shape[0], seq_length)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        player_ids: Optional[torch.LongTensor] = None,
        chat_type_ids: Optional[torch.LongTensor] = None,
        team_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        # Inputs_embeds: we already have the input embeddings, so no input_ids

        # Get the length of this sentence sequence
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # If we do not pass any position_ids, we use our buffered generated position ids.
        if position_ids is None:
            position_ids = self.position_ids[:,past_key_values_length: seq_length + past_key_values_length]

        token_type_ids = self.__set_default_if_none(attribute_ids = token_type_ids,
                                                    attribute_name = "token_type_ids",
                                                    seq_length = seq_length,
                                                    input_shape = input_shape)

        player_ids = self.__set_default_if_none(attribute_ids= player_ids,
                                        attribute_name="player_ids",
                                        seq_length=seq_length,
                                        input_shape=input_shape)

        chat_type_ids = self.__set_default_if_none(attribute_ids= chat_type_ids,
                                           attribute_name="chat_type_ids",
                                           seq_length=seq_length,
                                           input_shape=input_shape)

        team_ids = self.__set_default_if_none(attribute_ids = team_ids,
                                      attribute_name = "team_ids",
                                      seq_length=seq_length,
                                      input_shape=input_shape)



        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        player_ids_embeddings = self.player_ID_embeddings(player_ids)
        chat_type_embeddings = self.chat_type_embeddings(chat_type_ids)
        team_ids_embeddings = self.team_type_embeddings(team_ids)

        embeddings = inputs_embeds + token_type_embeddings \
                   + player_ids_embeddings \
                   + chat_type_embeddings \
                   + team_ids_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertWithSpeakerSegmentationPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertWithSpeakerSegmentationConfig
    load_tf_weights = modeling_bert.load_tf_weights_in_bert
    base_model_prefix = "bert-with-speaker-segmentation"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, modeling_bert.BertEncoder):
            module.gradient_checkpointing = value


class BertWithSpeakerSegmentationModel(BertWithSpeakerSegmentationPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertWithSpeakerSegmentationEmbeddings(config)
        self.encoder =  modeling_bert.BertEncoder(config)
        self.pooler = modeling_bert.BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def __replace_if_none(self, value, replacement):
        return value if value is not None else replacement

    def __set_default_if_none(self,
                              attribute_ids: Optional[torch.LongTensor],
                              attribute_name: str,
                              seq_length: int,
                              input_shape: Tuple[int],
                              device
                              ) -> torch.LongTensor:
        '''
        Helper function to set the default attribute if provided is `None`.
        We use buffer already registered (token_type_id in modeling_bert.BertEmbeddings)
        '''
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664

        # Return original if already set
        if attribute_ids is not None:
            return attribute_ids

        # If we no attribute, we just set a set of zero.
        if not hasattr(self, attribute_name):
            return torch.zeros(input_shape,
                               dtype=torch.long,
                               device=device)

        # Return expanded from buffer
        buffered_attribute_ids = self.embeddings.token_type_ids[:, :seq_length]
        return buffered_attribute_ids.expand(input_shape[0], seq_length)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        player_ids: Optional[torch.LongTensor] = None,
        chat_type_ids: Optional[torch.LongTensor] = None,
        team_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = self.__replace_if_none(output_attentions, self.config.output_attentions)
        output_hidden_states = self.__replace_if_none(output_hidden_states, self.config.output_hidden_states)
        return_dict = self.__replace_if_none(return_dict, self.config.use_return_dict)

        use_cache = self.config.is_decoder and self.__replace_if_none(use_cache, self.config.use_cache)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device)

        # Updated to include player_ids, chat_type_ids, team_ids
        token_type_ids = self.__set_default_if_none(attribute_ids = token_type_ids,
                                                    attribute_name = "token_type_ids",
                                                    seq_length = seq_length,
                                                    input_shape = input_shape,
                                                    device = device)

        player_ids = self.__set_default_if_none(attribute_ids= player_ids,
                                                attribute_name="player_ids",
                                                seq_length = seq_length,
                                                input_shape = input_shape,
                                                device = device)

        chat_type_ids = self.__set_default_if_none(attribute_ids= chat_type_ids,
                                                   attribute_name="chat_type_ids",
                                                   seq_length = seq_length,
                                                   input_shape = input_shape,
                                                   device = device)

        team_ids = self.__set_default_if_none(attribute_ids = team_ids,
                                              attribute_name = "team_ids",
                                              seq_length = seq_length,
                                              input_shape = input_shape,
                                              device = device)


        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device=device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Changed to include player_ids, chat_type_ids and team_ids

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            player_ids = player_ids,
            chat_type_ids = chat_type_ids,
            team_ids = team_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertWithSpeakerSegmentationForTokenClassification(BertWithSpeakerSegmentationPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert_with_speaker_segmentation = BertWithSpeakerSegmentationModel(config, add_pooling_layer=False)
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        player_ids: Optional[torch.LongTensor] = None,
        chat_type_ids: Optional[torch.LongTensor] = None,
        team_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Update to include player_ids, chat_type_ids and team_ids
        outputs = self.bert_with_speaker_segmentation(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            player_ids = player_ids,
            chat_type_ids = chat_type_ids,
            team_ids = team_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
