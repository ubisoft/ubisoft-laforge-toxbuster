from dataclasses import dataclass
import functools
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import torch
from loguru import logger
from transformers import AutoTokenizer

from src.Data.Dataset import ToxDataReturn


class Collate(ABC):
    '''
    Collates the data and assigns the correct label.
    '''
    @abstractmethod
    def __call__(self, batch: List[ToxDataReturn]) -> Dict[str, torch.Tensor]:
        '''
        Function to collate the data.

        Parameters:
        ----------
        batch: List[ToxDataReturn]
            A batch of our dataset.

        Returns:
        --------
        A dictionary containing the tokenized_inputs and labels.
        '''
        return NotImplementedError

class SentenceCollate(Collate):
    '''
    Collates the data from the tokenizer.
    Assigns the correct label.
    Provides the correct prediction mask.

    Attributes:
    -----------
    tokenizer_name: str
        String name of the AutoTokenizer.
    tokenizer: AutoTokenizer
        This is initialized using `tokenizer_name`.
    max_token_length: int
        Max Length for an instance. This is one of the input size of the model.
    train: bool
        Whether this data collation is for train / not.
    truncation_strategy: str
        Truncation strategy provided by HuggingFace
    include_context: bool
        Whether to include the context or not.

    '''

    def __init__(self, tokenizer: str,
                 max_token_length: int,
                 train: bool = True,
                 include_context: bool = True,
                 truncation_strategy: str = "longest_first",
                 truncation_side: str = None,
                 special_tokens: Dict = None,
                 ):
        '''
        Initializes the data collation function.

        Parameter:
        ----------
        tokenizer: str
            Name of the pretrained autotokenizer.
        max_token_length: int
            Max Length for an instance.
        train: bool
            Whether this collation is for train / not.
        include_context: bool
            Whether to include context or not.
        '''
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_token_length = max_token_length
        self.train = train
        self.include_context = include_context
        self.truncation_strategy = truncation_strategy

        if (special_tokens is not None):
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            logger.info(f"Added {num_added} tokens to the tokenizer!")
        if (truncation_side is not None):
            self.tokenizer.truncation_side = truncation_side

    def __call__(self, batch: List[ToxDataReturn]) -> Dict[str, torch.Tensor]:
        '''
        Function to collate the data.

        Parameters:
        ----------
        batch: List[ToxDataReturn]
            A batch of our dataset.

        Returns:
        --------
        A dictionary containing the tokenized_inputs and labels.
        '''
        # Extract context and text from the batch.
        context = list(
            map(lambda batch_instance: batch_instance.context, batch))
        text = list(map(lambda batch_instance: batch_instance.text, batch))

        # Context is first because it comes before the current chat line.
        # Truncation & padding strategy https://huggingface.co/docs/transformers/pad_truncation

        if (not self.include_context):
            context = list(map(lambda batch_instance: [], batch))

        tokenized_inputs = self.tokenizer(text=context,
                                          text_pair=text,
                                          padding="longest",
                                          truncation=self.truncation_strategy,
                                          max_length=self.max_token_length,
                                          return_offsets_mapping=True,
                                          is_split_into_words=True,
                                          return_tensors="pt"
                                          )

        # Return tokenized inputs without any labels
        if (self.train):
            labels = list(
                map(lambda batch_instance: self.get_most_frequent_toxic_label(batch_instance.label), batch))
        else:
            labels = [1 for _ in text]

        prediction_masks = [1 for _ in text]

        aligned_labels = torch.LongTensor(labels)
        prediction_masks = torch.LongTensor(prediction_masks)

        tokenized_inputs.pop("offset_mapping")
        return dict(tokenized_inputs, **{"labels": aligned_labels, "prediction_mask": prediction_masks})

    @staticmethod
    def get_most_frequent_toxic_label(labels: List[int]):
        '''
        Uses the most frequent toxic label. If tied, use most toxic.
        '''
        labels = np.array(labels)
        if np.any(labels > 0):
            out = stats.mode(labels[labels > 0])[0][0]
        else:
            out = 0
        return out


@dataclass
class NeededAlignment:
    '''
    Struct to keep the alignment info
    '''
    text_part: list
    context_part: list
    special_default_value: object

class TokenCollate(Collate):
    '''
    Collates the data from the tokenizer.
    Assigns the correct label.
    Provides the correct prediction mask.

    Attributes:
    -----------
    tokenizer_name: str
        String name of the AutoTokenizer.
    tokenizer: AutoTokenizer
        This is initialized using `tokenizer_name`.
    max_token_length: int
        Max Length for an instance. This is one of the input size of the model.
    train: bool
        Whether this data collation is for train / not.
    truncation_strategy: str
        Truncation strategy provided by HuggingFace
    include_context: bool
        Whether to include the context or not.

    '''

    def __init__(self, tokenizer: str,
                 max_token_length: int,
                 train: bool = True,
                 include_context: bool = True,
                 subtoken_label_strategy: str = "skip",
                 truncation_strategy: str = "longest_first",
                 truncation_side: str = None,
                 special_tokens: Dict = None,
                 with_speaker_segmentation: bool = False,
                 ):
        '''
        Initializes the data collation function.

        Parameter:
        ----------
        tokenizer: str
            Name of the pretrained autotokenizer.
        max_token_length: int
            Max Length for an instance.
        train: bool
            Whether this collation is for train / not.
        include_context: bool
            Whether to include context or not.
        subtoken_label_strategy: str
            Defaults to `skip`.
        with_speaker_segmentation: bool
            Defaults to False. Whether to include speaker segmentation or not.
        '''
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, add_prefix_space=True)
        self.max_token_length = max_token_length
        self.train = train
        self.include_context = include_context
        self.subtoken_label_strategy = subtoken_label_strategy
        self.truncation_strategy = truncation_strategy
        self.with_speaker_segmentation = with_speaker_segmentation

        if (special_tokens is not None):
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            logger.info(f"Added {num_added} tokens to the tokenizer!")
        if (truncation_side is not None):
            self.tokenizer.truncation_side = truncation_side

    def __call__(self, batch: List[ToxDataReturn]) -> Dict[str, torch.Tensor]:
        '''
        Function to collate the data.

        Parameters:
        ----------
        batch: List[ToxDataReturn]
            A batch of our dataset.

        Returns:
        --------
        A dictionary containing the tokenized_inputs and labels.
        '''
        # Extract context and text from the batch.
        context = list(
            map(lambda batch_instance: batch_instance.context, batch))
        text = list(map(lambda batch_instance: batch_instance.text, batch))

        # Context is first because it comes before the current chat line.
        # Truncation & padding strategy https://huggingface.co/docs/transformers/pad_truncation

        if (not self.include_context):
            context = list(map(lambda batch_instance: [], batch))

        tokenized_inputs = self.tokenizer(text=context,
                                          text_pair=text,
                                          padding="longest",
                                          truncation=self.truncation_strategy,
                                          max_length=self.max_token_length,
                                          return_offsets_mapping=True,
                                          is_split_into_words=True,
                                          return_tensors="pt"
                                          )

        # Align provided labels to the correct token for "full_text"
        result = {
            "labels": [],
            "prediction_mask": []
        }

        if (self.with_speaker_segmentation):
            result["chat_type_ids"] = []
            result["player_ids"] = []
            result["team_ids"] = []

        troublesome_rows = []

        ignore_row_labels = [-100 for _ in range(0, self.max_token_length)]
        ignore_row_mask = [0 for _ in range(0, self.max_token_length)]

        for i, (instance_offset_mapping, instance) in enumerate(zip(tokenized_inputs["offset_mapping"], batch)):
            try:
                aligned = self.align_single_instance(instance_offset_mapping,
                                                     instance,
                                                     subtoken_label_strategy = self.subtoken_label_strategy,
                                                     with_speaker_segmentation = self.with_speaker_segmentation)
                for k, v in aligned.items():
                    result[k].append(v)

            except:
                # TODO: CHECK IF THERE IS A BETTER WAY TO DO THIS, Skip troublesome lines for now... most likely not English
                token_length = instance_offset_mapping.size()[0]
                troublesome_rows.append(i)

                for k in result.keys():
                    if k == "labels":
                        result[k].append(ignore_row_labels[:token_length])
                    else:
                        result[k].append(ignore_row_mask[:token_length])

                logger.error(instance)
                logger.error(instance_offset_mapping.size())
                logger.error(instance_offset_mapping.tolist())


        if (len(troublesome_rows) > 0):
            logger.error(f"Troublesome rows: {troublesome_rows}")

        # Cast to LongTensor
        for i in result:
            result[i] = torch.LongTensor(result[i])

        tokenized_inputs.pop("offset_mapping")
        return dict(tokenized_inputs, **result)

    @staticmethod
    def align_single_instance(instance_offset_mapping: Tuple[List[int], List[int]],
                              instance: ToxDataReturn,
                              subtoken_label_strategy: str,
                              with_speaker_segmentation: bool):
        '''
        Helper method to assign the right labels to each instance.
        Useful link: https://huggingface.co/docs/transformers/tasks/token_classification

        Some rules from the link:
            > Map tokens to their corresponding word with the word_ids method.

            > Assign label -100 to special tokens [CLS] and [SEP] so loss
            function ignores them.

            > Only label the first token of a given word. Assign -100 to other subtokens from the same word.

        For our case currently:
            > Set all tokens in our context as -100.

        Parameters:
        -----------
        instance_offset_mapping: list[tuple(int, int)]
            One (instance) of the offset mapping from the tokenizer.
        instance: ToxDataReturn
            The instance of our dataset.
        subtoken_label_strategy: str
            Should be either `skip` or `inherit`. `Skip` gives them labels -100. `Inherit` gives them labels of the head token.
        with_speaker_segmentation: bool
            Whether there is speaker segmentation info or not.

        Returns:
        --------
        The labels for each token for this instance.

        Example:
        -------
        Sample Full Text of `["don't"]` and context of `[]` can be tokenized
        to `["don", "t"]`. The offset mapping wil be like below. Note, for
        text_pair, the tokenization will result in
        `<CLS> [context] <SEP> [full_text] <SEP>`.
        '''
        result = {"labels": []}
        prediction_mask = []

        special_tokens_seen = 0
        context_index = 0
        text_index = 0

        need_alignments = {
            "labels": NeededAlignment(text_part = instance.label if instance.label is not None else np.ones(len(instance.text)),
                                      context_part = instance.context_label if instance.context_label is not None else ([-100] * len(instance_offset_mapping)),
                                      special_default_value = -100),
        }

        if (with_speaker_segmentation):
            need_alignments["chat_type_ids"] = NeededAlignment(text_part=instance.chat_type if instance.chat_type is not None else np.zeros(len(instance.text)),
                                                               context_part=instance.context_chat_types if instance.context_chat_types is not None else np.zeros(len(instance_offset_mapping)),
                                                               special_default_value=0)
            result["chat_type_ids"] = []

            need_alignments["player_ids"] = NeededAlignment(text_part=instance.player_id if instance.player_id is not None else np.zeros(len(instance.text)),
                                                            context_part=instance.context_player_ids if instance.context_player_ids is not None else np.zeros(len(instance_offset_mapping)),
                                                            special_default_value=0)
            result["player_ids"] = []

            need_alignments["team_ids"] = NeededAlignment(text_part=instance.team_id if instance.team_id is not None else np.zeros(len(instance.text)),
                                                          context_part=instance.context_team_ids if instance.context_team_ids is not None else np.zeros(len(instance_offset_mapping)),
                                                          special_default_value=0)
            result["team_ids"] = []

        for (token_offset_start, token_offset_end) in instance_offset_mapping:

            # Case 1: Assign special default value to special tokens: [CLS], [SEP], PADDING
            #         We do not predict on special tokens.
            if token_offset_end == 0:

                for k in need_alignments.keys():
                    result[k].append(need_alignments[k].special_default_value)

                prediction_mask.append(0)
                special_tokens_seen += 1

            else:
                # Case 2: We are looking at the context
                if special_tokens_seen < 2:

                    for k in need_alignments.keys():
                        # Case 2.1: Head of the token in context
                        if (token_offset_start == 0):
                            aligned_value = need_alignments[k].context_part[context_index]

                        # Case 2.2: Subtoken in context
                        else:
                            if (subtoken_label_strategy == "inherit"):
                                subtoken_index = context_index - 1
                                aligned_value = need_alignments[k].context_part[subtoken_index]
                            else:
                                aligned_value = need_alignments[k].special_default_value
                        result[k].append(aligned_value)

                    prediction_mask.append(0)
                    if (token_offset_start == 0):
                        context_index += 1

                # Case 3: Looking at full_text
                else:
                    # Case 3.1 Head of token of full text
                    for k in need_alignments.keys():

                        if (token_offset_start == 0):
                            aligned_value = need_alignments[k].text_part[text_index]
                        # Case 3.2 Subtoken of full text
                        else:
                            if (subtoken_label_strategy == "inherit"):
                                subtoken_index = text_index - 1
                                aligned_value = need_alignments[k].text_part[subtoken_index]
                            else:
                                aligned_value = need_alignments[k].special_default_value

                        result[k].append(aligned_value)

                    # Calculate prediction mask &
                    if (token_offset_start == 0):
                        prediction_mask.append(1)
                        text_index += 1
                    else:
                        prediction_mask.append(0)

        result["prediction_mask"] = prediction_mask
        return result



class TokenCollate_v2(TokenCollate):
    '''
    Collates the data from the tokenizer.
    Assigns the correct label.
    Provides the correct prediction mask.

    Attributes:
    -----------
    tokenizer_name: str
        String name of the AutoTokenizer.
    tokenizer: AutoTokenizer
        This is initialized using `tokenizer_name`.
    max_token_length: int
        Max Length for an instance. This is one of the input size of the model.
    train: bool
        Whether this data collation is for train / not.
    truncation_strategy: str
        Truncation strategy provided by HuggingFace
    include_context: bool
        Whether to include the context or not.

    '''
    def __call__(self, batch: List[ToxDataReturn]) -> Dict[str, torch.Tensor]:
        '''
        Function to collate the data.

        Parameters:
        ----------
        batch: List[ToxDataReturn]
            A batch of our dataset.

        Returns:
        --------
        A dictionary containing the tokenized_inputs and labels.
        '''

        # Extract context and text from the batch.
        context = list(
            map(lambda batch_instance: batch_instance.context, batch))
        text = list(map(lambda batch_instance: batch_instance.text, batch))

        # Context is first because it comes before the current chat line.
        # Truncation & padding strategy https://huggingface.co/docs/transformers/pad_truncation

        if (not self.include_context):
            context = list(map(lambda batch_instance: [], batch))

        tokenized_inputs = self.tokenizer(text=context,
                                          text_pair=text,
                                          padding="longest",
                                          truncation=self.truncation_strategy,
                                          max_length=self.max_token_length,
                                          is_split_into_words=True,
                                          return_tensors="pt"
                                          )
        labels = []
        prediction_masks = []

        ignored_context_labels = [-100] * self.max_token_length

        for i, instance in enumerate(batch):
            sequence_ids = tokenized_inputs.sequence_ids(batch_index = i)
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            previous_word_idx = None
            prediction_mask_ids = []
            label_ids = []

            for seq_idx, word_idx in zip(sequence_ids, word_ids):

                # We only predict on the text_pair (seq_idx == 1 not 0).
                # We do not predict on special tokens.
                # We do not predict on the second part of the word if not inheriting
                if (seq_idx == 1 and
                    word_idx is not None and
                    (word_idx != previous_word_idx or
                    (word_idx == previous_word_idx and self.subtoken_label_strategy == "inherit"))):
                    prediction_mask_ids.append(1)
                else:
                    prediction_mask_ids.append(0)
                # prediction_mask_ids.append(1 if seq_idx == 1 else 0)

                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    continue

                # Point to context / label
                context_label = instance.context_label
                if instance.context_label is None:
                    context_label = ignored_context_labels
                label = context_label if seq_idx == 0 else instance.label
                if label is None:
                    label = ignored_context_labels

                # We set the label for the first token of each word.
                if word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_id = -100
                    if self.subtoken_label_strategy == "inherit":
                        label_id = label[word_idx]
                    label_ids.append(label_id)

                previous_word_idx = word_idx

            prediction_masks.append(prediction_mask_ids)
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["prediction_mask"] = prediction_masks

        for i in tokenized_inputs.keys():
            tokenized_inputs[i] = torch.LongTensor(tokenized_inputs[i])

        return tokenized_inputs


class TokenCollateCustomTruncation(Collate):
    '''
    Collates the data from the tokenizer.
    Assigns the correct label.
    Provides the correct prediction mask.

    This collation will result in data in the format of:
        <CLS> [CONTEXT] <SEP> [FULL TEXT] <SEP>

    Notes:
    - Data will always be of max_token_length
    - Pad length is currently fixed to follow "longest".
    - Padding is currently fixed to be on the right side after <SEP>. Left side will be before <CLS>
    - Truncation is fixed. It follows the logic of:
        * If full_text is longer than max_token_length, truncate starting from the right of full_text.
        * If context + full_text is longer than max_token_length, truncate from the left of context.


    Attributes:
    -----------
    tokenizer_name: str
        String name of the AutoTokenizer.
    tokenizer: AutoTokenizer
        This is initialized using `tokenizer_name`.
    max_token_length: int
        Max Length for an instance. This is one of the input size of the model.
    train: bool
        Whether this data collation is for train / not.
    include_context: bool
        Whether to include the context or not.
    '''

    def __init__(self, tokenizer: str,
                 max_token_length: int,
                 train: bool = True,
                 include_context: bool = True,
                 context_truncation_side: str = "left",
                 subtoken_label_strategy: str = "skip",
                 padding_longest_in_batch: bool = True
                 ):
        '''
        Initializes the data collation function.

        Parameter:
        ----------
        tokenizer: str
            Name of the pretrained autotokenizer.
        max_token_length: int
            Max Length for an instance.
        train: bool
            Defaults to True. Whether this collation is for train / not.
        include_context: bool
            Defaults to True. Whether to include context or not.
        context_truncation_side: str
            Defaults to left. Can be either left or right.
        subtoken_label_strategy: str
            Defaults to `ignore`. Can be in [`skip`, `inherit`].

            `skip` gives the subtoken the label of -100.
                i.e. "Don't" with label of 1 => ["Don", "t"] with labels of [1, -100]

            `inherit` gives the subtoken the same label as the token.
                i.e. "Don't" with label of 1 => ["Don", "t"] with labels of [1, 1]
        padding_longest_in_batch: bool
            Whether to shorten padding to longest in batch.

        '''
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, truncation_side="right")
        self.context_truncation_side = context_truncation_side
        self.context_tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, truncation_side=context_truncation_side)
        self.max_token_length = max_token_length
        self.train = train
        self.include_context = include_context
        self.special_token_outputs = self.tokenizer(text=[],
                                                    is_split_into_words=True,
                                                    return_offsets_mapping=True,
                                                    max_length=3,
                                                    padding="max_length")
        self.subtoken_label_strategy = subtoken_label_strategy
        self.padding_longest_in_batch = padding_longest_in_batch

    def __call__(self, batch: List[ToxDataReturn]) -> Dict[str, torch.Tensor]:
        '''
        Function to collate the data.

        Parameters:
        ----------
        batch: List[ToxDataReturn]
            A batch of our dataset.

        Returns:
        --------
        A dictionary containing the tokenized_inputs, labels and prediction_mask
        '''
        final_dict = {}

        max_token_length_without_paddding = 0

        # Initialize final_dict
        empty_output, _ = self.tokenize_single_instance(ToxDataReturn(context=[],
                                                                      text=[],
                                                                      label=[],
                                                                      context_label=[]))
        for k in empty_output:
            final_dict[k] = []

        # Tokenize each instance & keep track of token length
        troublesome_rows = []
        for i, instance in enumerate(batch):

            try:
                if (not self.include_context):
                    instance.context = []

                output, token_length_without_padding = self.tokenize_single_instance(
                    instance)

                # Keep track of maximum token length
                max_token_length_without_paddding = max(
                    max_token_length_without_paddding, token_length_without_padding)

                # Append output to final output
                for k, v in output.items():
                    final_dict[k].append(v)
            except Exception as e:
                troublesome_rows.append(i)

                # Change troublesome rows to an empty output.
                for k in final_dict:
                    final_dict[k].append(empty_output[k])

        if (len(troublesome_rows) > 0):
            logger.error(f"Troublesome rows: {troublesome_rows}")

        # Mask of padding since it is currently fixed at "longest" of the batch
        if (self.padding_longest_in_batch):
            padding_mask = torch.arange(0,  max_token_length_without_paddding)

            for k, v in final_dict.items():
                new_v = torch.LongTensor(v)
                new_v = torch.index_select(new_v, 1, padding_mask)
                final_dict[k] = new_v

        return final_dict

    @staticmethod
    def align_label_prediction_mask_to_offset_mapping(offset_mapping: List[Tuple[int, int]],
                                                      labels: Optional[List[int]],
                                                      subtoken_label_strategy: str,
                                                      ) -> Tuple[List[int], List[int]]:
        '''
        Given the offset mapping of the text without special tokens added, return the proper position for the label and prediction mask.

        Parameter:
        ----------
        offset_mapping: List[Tuple]
            Mapping of the token offset
        labels: List[int]
            List of labels for each token. If None, tokens are labelled as -100.
        subtoken_label_strategy: str
           Should be either `skip` or `inherit`.

        Returns:
        --------
        A tuple in the format of aligned_labels, prediction_mask
        '''
        aligned_labels = []
        prediction_mask = []
        label_index = 0

        if labels == None:
            labels = [-100] * len(offset_mapping)

        for (token_offset_start, _) in offset_mapping:
            if token_offset_start > 0:

                # Default is giving -100 to subtokens.
                # If `inherit`, we give subtokens the same label as the head.
                if (subtoken_label_strategy == "inherit"):
                    # Subtoken is the last token in list
                    subtoken_label_index = label_index - 1
                    aligned_labels.append(labels[subtoken_label_index])
                else:
                    aligned_labels.append(-100)

                prediction_mask.append(0)
            else:
                aligned_labels.append(labels[label_index])
                prediction_mask.append(1)
                label_index += 1

        return aligned_labels, prediction_mask

    @staticmethod
    def align_single_instance_labels_predictions(context_offset_mapping: List[Tuple[int, int]],
                                                 context_labels: Optional[List[int]],
                                                 context_start: int,
                                                 context_end: int,
                                                 full_text_offset_mapping: List[Tuple[int, int]],
                                                 full_text_labels: Optional[List[int]],
                                                 pad_length: int,
                                                 subtoken_label_strategy: str) -> Tuple[List[int], List[int]]:
        '''
        Aligns an instance with context and full text the full label and prediction mask.

        Parameter:
        ----------
        context_offset_mapping: List[Tuple]
            Mapping of the token offset on the context
        context_labels: List[int]
            List of labels for each token in the context.
        context_length: int
            Number of tokens created from context.
        full_text_offset_mapping: List[Tuple]
            Mapping of the token offset on the full text
        full_text_labels: List[int]
            List of labels for each token in the full text.
        pad_length: int
            Number of pad tokens.
        subtoken_label_strategy: str
           Should be either `skip` or `inherit`.

        Returns:
        --------
        A tuple in the format of aligned_labels, prediction_mask
        '''
        aligned_context_labels, _ = TokenCollateCustomTruncation.align_label_prediction_mask_to_offset_mapping(
            context_offset_mapping, context_labels, subtoken_label_strategy)
        aligned_full_text_labels, aligned_full_text_prediction_mask = TokenCollateCustomTruncation.align_label_prediction_mask_to_offset_mapping(
            full_text_offset_mapping, full_text_labels, subtoken_label_strategy)

        # Build seq based on <CLS> [Context ...] <SEQ> [ Full Text ... ] <SEP> <PAD ... >
        labels = [-100] + aligned_context_labels[context_start: context_end] + [-100] + \
            aligned_full_text_labels + [-100] * (pad_length + 1)
        prediction_mask = [0] * (context_end - context_start + 2) + \
            aligned_full_text_prediction_mask + [0] * (pad_length + 1)

        return labels, prediction_mask

    def tokenize_single_instance(self, instance: ToxDataReturn):
        '''
        Given an instance of ToxDataReturn, tokenize the context and text and provide the aligned labels and prediction mask.

        Parameter:
        ----------
        instance: ToxDataReturn
            One row of our dataset.

        Returns:
        ----------
        (output, token_length_without_padding)
        output is a dictionary containing the input_ids, labels, prediction_mask and any other keys from the auto tokenizer.

        '''

        full_text_output = self.tokenizer(text=instance.text,
                                          is_split_into_words=True,
                                          return_offsets_mapping=True,
                                          add_special_tokens=False,
                                          max_length=self.max_token_length,
                                          truncation=True)

        # Deduct for <CLS> <SEP> <SEP>
        max_length = self.max_token_length - 3

        # Calculate lengths to build correct sequence
        full_text_length = min(max_length, len(full_text_output["input_ids"]))
        allowed_context_length = max_length - full_text_length

        context = [] if allowed_context_length == 0 else instance.context

        context_output = self.context_tokenizer(text=context,
                                                is_split_into_words=True,
                                                return_offsets_mapping=True,
                                                add_special_tokens=False,
                                                max_length=self.max_token_length,
                                                truncation=True)
        context_label = instance.context_label

        if (self.context_truncation_side == "right"):
            context_start_index = 0
            context_end_index = min(
                allowed_context_length, len(context_output["input_ids"]))
        else:
            context_start_index = max(
                0, len(context_output["input_ids"]) - allowed_context_length)
            context_end_index = len(context_output["input_ids"])

        context_length = context_end_index - context_start_index
        pad_length = max_length - full_text_length - context_length

        final_output = {}

        for k, v in full_text_output.items():

            if (k == "offset_mapping"):
                continue

            cls_part = self.special_token_outputs[k][0]
            sep_part = self.special_token_outputs[k][1]
            pad_part = self.special_token_outputs[k][2]

            context_part = context_output[k][context_start_index:context_end_index]

            # Build sequence based on <CLS> [Context ...] <SEQ> [ Full Text ... ] <SEP> <PAD ... >
            seq = [cls_part] + context_part + [sep_part] + \
                v[:full_text_length] + [sep_part] + [pad_part] * pad_length

            final_output[k] = seq

        l, m = TokenCollateCustomTruncation.align_single_instance_labels_predictions(
            context_offset_mapping=context_output["offset_mapping"],
            context_labels=context_label,
            context_start=context_start_index,
            context_end=context_end_index,
            full_text_offset_mapping=full_text_output["offset_mapping"][:full_text_length],
            full_text_labels=instance.label,
            pad_length=pad_length,
            subtoken_label_strategy=self.subtoken_label_strategy)

        final_output["labels"], final_output["prediction_mask"] = l, m

        token_length_without_padding = context_length + full_text_length + 3
        return final_output, token_length_without_padding
