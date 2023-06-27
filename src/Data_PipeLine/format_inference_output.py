from itertools import groupby
import pandas as pd


def format_output(df):
    """Transform word-based model output to span-based output, possibly adding lines in the process

    Expects a data frame with at least
    "matchid"
    "line_index"
    "full_text_arr": message in array of words
    "label_arr": array of labels corresponding to each word, length must be equal.
    """
    data_dict = {
        'match_id': [], 'line_index': [], 'message': [], 'start_string_index': [],
        'end_string_index': [], 'category_id': []
    }
    df = df.loc[:, ["matchid", "line_index", "full_text_arr", "label_arr"]]
    for _, matchid, line_index, full_text, labels in df.itertuples():
        # print(matchid, line_index, full_text, labels)
        label_list = []
        word_index = 0
        character_index = 0
        for label, group in groupby(labels):
            group_len = len(list(group))
            end_word_index = word_index + group_len
            selection_length = sum(len(word) for word in full_text[word_index:end_word_index]) + group_len - 1
            selection_start = character_index
            selection_stop = character_index + selection_length
            if label != 0:
                label_list.append([label, (selection_start, selection_stop)])
            # print(label, group_len, word_index, end_word_index, selection_length, selection_start, selection_stop)
            word_index = end_word_index
            character_index = selection_stop + 1
        pasted_text = ' '.join(full_text)
        if not label_list:
            data_dict['match_id'].append(matchid)
            data_dict['line_index'].append(line_index)
            data_dict['message'].append(pasted_text)
            data_dict['start_string_index'].append(0)
            data_dict['end_string_index'].append(len(pasted_text))
            data_dict['category_id'].append(0)
        else:
            for label, (start_char_index, end_char_index) in label_list:
                data_dict['match_id'].append(matchid)
                data_dict['line_index'].append(line_index)
                data_dict['message'].append(pasted_text)
                data_dict['start_string_index'].append(start_char_index)
                data_dict['end_string_index'].append(end_char_index)
                data_dict['category_id'].append(label)

    return pd.DataFrame(data_dict)
