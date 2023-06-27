from typing import List

import numpy as np

class LabelStrategy:
    '''
    Base Label Strategy
    '''

    def __call__(self,
                 text: str,
                 start_indices: List[int],
                 end_indices: List[int],
                 tags: List[int]) -> List[int]:
        return NotImplemented

class LabelTokensAsIs(LabelStrategy):
    '''
    Label Tokens As Is (No change from DataFrame)
    '''
    def __call__(self,
                 text: str,
                 start_indices: List[int],
                 end_indices: List[int],
                 tags: List[int]) -> List[int]:
        '''
        Label tokens as provided.
        '''
        return label_raw_text(text, start_indices, end_indices, tags, tagging_system="AsIs")

class LabelTokensStatic(LabelStrategy):
    '''
    Ignore tokens in DataFrame and set to a static label.
    '''
    def __init__(self,
                 static_label: int = 0) -> None:
        super().__init__()
        self.static_label = static_label


    def __call__(self,
                 text: str,
                 start_indices: List[int],
                 end_indices: List[int],
                 tags: List[int]) -> List[int]:
        '''
        Provide a dummy label for each token.
        '''
        return [self.static_label for t in text.split()]

class LabelTokensAdjustedForContext(LabelStrategy):
    '''
    Label tokens same as DataFrame but adjusted for context label.

    Assumption:
        Context_label takes 0.
        Original labels => += 1
    '''
    label_as_is = LabelTokensAsIs()

    def __call__(self,
                text: str,
                start_indices: List[int],
                end_indices: List[int],
                tags: List[int]):
        '''
        Label tokens based on above assumptions.
        '''
        labels = self.label_as_is(text, start_indices, end_indices, tags)
        return [label + 1 for label in labels]

class LabelTokensWithBIO(LabelStrategy):
    '''
    Label Tokens using BIO tagging system.
    '''
    def __call__(self,
                 text: str,
                 start_indices: List[int],
                 end_indices: List[int],
                 tags: List[int]) -> List[int]:
        '''
        Label tokens as provided.
        '''
        return label_raw_text(text, start_indices, end_indices, tags, tagging_system="BIO")

class LabelTokensWithBILOU(LabelStrategy):
    '''
    Label Tokens using BILOU tagging system.
    '''
    def __call__(self,
                 text: str,
                 start_indices: List[int],
                 end_indices: List[int],
                 tags: List[int]) -> List[int]:
        '''
        Label tokens as provided.
        '''
        return label_raw_text(text, start_indices, end_indices, tags, tagging_system="BILOU")


def get_selected_text_tagging(selected: List[str],
                              tag: int,
                              tagging_system: str):
    '''
    Using provided text tagging system, tag the selected text.

    Parameter:
    ------------
    selected:
        The list of text that is selected as toxic.
    tag:
        Integer toxic category
    tagging_system:
        Tagging system must be in "AsIs", "BIO", "BILOU"
    '''

    if (tagging_system == "AsIs"):
        return [tag for x in selected]

    if (tagging_system == "BIO"):
        # Map tag into the following format
        # 0 -> 0
        # 1 -> 1  B-1
        #   -> 2  I-1
        # 2 -> 3  B-2
        # 3 -> 5  B-3
        tag = 2 * tag - 1 if tag > 0 else 0
        labels = []
        if (len(selected) >= 1):
            labels.append(tag)          # B - tag
            for _ in selected[1:]:
                labels.append(tag + 1)  # I - tag

        return labels

    if (tagging_system == "BILOU"):
        # Map tag into the following format
        # 0 -> 0
        # 1 -> 1  B-1
        #   -> 2  I-1
        #   -> 3  L-1
        #   -> 4  U-1
        # 2 -> 5  B-2
        # 3 -> 9  B-3
        tag = 4 * tag - 3 if tag > 0 else 0
        labels = []
        if len(selected) == 1:
            labels = [tag + 3] # U-tag
        if (len(selected) > 1):
            labels.append(tag)          # B-tag
            for _ in selected[1:-1]:
                labels.append(tag + 1)  # I-tag
            labels.append(tag + 2)      # L-tag

        return labels

    raise Exception(f"Provided tagging_system {tagging_system} is not in ['AsIs', 'BIO', 'BILOU']")

def label_raw_text(text: str,
                   start_indices: List[int],
                   end_indices: List[int],
                   tags: List[int],
                   tagging_system: str) -> List[int]:
    '''
    Labels the raw text provided the start_indices, end_indices, tags and the tagging system.
    '''
    # If tag_col is [nan] or [0,0]
    # If start_indices / end_indices is [nan]
    if (len(tags) == 0 or
        (len(tags) == 1 and np.isnan(tags[0])) or
        (len(start_indices) == 1 and np.isnan(start_indices[0])) or
        (len(end_indices) == 1 and np.isnan(end_indices[0]))):
        return [0 for t in text.split()]

    label = []
    prev_start = 0

    for i, tag in enumerate(tags):

        if (np.isnan(start_indices[i])):
            start = 0
        else:
            start = int(start_indices[i])
        if (np.isnan(end_indices[i])):
            end = 0
        else:
            end = int(end_indices[i])
        tag = int(tag)

        before = text[prev_start:start].split()
        selected = text[start:end].split()

        for _ in before:
            label.append(0)
        label += get_selected_text_tagging(selected, tag, tagging_system)
        prev_start = end

    # take care of the last end of the text.
    after = text[prev_start:].split()
    for _ in after:
        label.append(0)

    return label