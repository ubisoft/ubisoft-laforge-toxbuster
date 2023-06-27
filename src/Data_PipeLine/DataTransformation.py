import math
from abc import ABC, abstractmethod
from tokenize import group
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import pandas as pd
from loguru import logger

from src.Data_PipeLine import LabelStrategy


class DataTransformation(ABC):
    """
    Transformations applied on the input pd.DataFrame
    """

    def __init__(self):
        """
        Initializes the DataTransformation.
        """
        return

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the transformation on the pd.DataFrame.

        Parameter:
        ----------
        df: pd.DataFrame
            The input DataFrame we are going to transform.

        Returns:
        ----------
        pd.DataFrame
            The transformed DataFrame
        """
        raise NotImplementedError


class BuildContext(DataTransformation):
    """
    Builds the context based on existing match id and line index.
    We can limit context based on `past_x_seconds`
    """

    def __init__(
        self,
        context_group_by: List[str] = None,
        timestamp_col: str = "relative_timestamp",
        past_x_seconds: float = math.inf,
        include_context_label: bool = False,
        chat_mode: str = "Full",
        include_new_line: bool = False,
        add_chat_type: bool = False,
        add_team_type: bool = False,
        add_player_id: bool = False,
        method_to_add_custom_context: str = "left",
    ):
        """
        Initializes the context builder.

        Parameter:
        ----------
        context_group_by: List[str]
            Defaults to None (["matchid"]). Defines what to group as match.
        timestamp_col: str
            Defaults to relative_timestamp. Defines what the timestamp column to use for past_x_seconds
        past_x_seconds: float
            Defaults to math.inf. The number of seconds before this chat line was sent to consider to include in the context.
        include_context_label: bool
            Defaults to False. Whether to also build the context labels
        chat_mode: str
            Defaults to `Full`. Must be in the one of `Full`, `Global`, `Team`, `Personal`.
        include_new_line: bool
            Defaults to False. Whether to add a new line between each chat line.
        add_chat_type: bool
            Defaults to False. Whether to add the chat type. How to add depends on `method_to_add_custom_context`.
        add_team_type: bool
            Defaults to False. Whether to add the team type. How to add depends on `method_to_add_custom_context`.
        add_player_id: bool
            Defaults to False. Whether to add the player ID type. How to add depends on `method_to_add_custom_context`.
        method_to_add_custom_context: bool
            Defaults to `left`. If at least one of `include_new_line, add_chat_type, add_team_type or add_player_id` is true, we will be adding custom context.

        """
        super().__init__()

        self.past_x_seconds = past_x_seconds
        self.context_group_by = (
            ["matchid"] if context_group_by is None else context_group_by
        )
        self.timestamp_col = timestamp_col
        self.chat_mode = chat_mode

        self.context_accumulating_columns = ["full_text"]
        self.context_new_column_names = ["context"]

        if include_context_label:
            self.context_accumulating_columns.append("label")
            self.context_new_column_names.append("context_label")

        self.complex_context = (
            include_new_line or add_chat_type or add_team_type or add_player_id
        )

        self.add_new_line_token = include_new_line

        self.add_chat_type = add_chat_type
        if add_chat_type:
            self.context_accumulating_columns.append("is_team_channel")
            self.context_new_column_names.append("is_team_channels")

        self.add_team_type = add_team_type
        if add_team_type:
            self.context_accumulating_columns.append("team")
            self.context_new_column_names.append("teams")

        self.add_player_id = add_player_id
        if add_player_id:
            self.context_accumulating_columns.append("profileid")
            self.context_new_column_names.append("profileids")
        if add_player_id and not add_team_type:
            self.context_accumulating_columns.append("team")
            self.context_new_column_names.append("teams")

        self.method_to_add_custom_context = method_to_add_custom_context

        logger.debug(f"Accumulating Columns: {self.context_accumulating_columns}")
        logger.debug(f"New Columns: {self.context_new_column_names}")

    def __str__(self):
        return f"BuildContext past ({self.past_x_seconds}) seconds - group By: {self.context_group_by}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the context for each chat line based on the previous chat lines given the time_filter.

        Parameter:
        ----------
        df: pd.DataFrame
            The input DataFrame we are going to transform.

        DataFrame Columns:
        -----------------------
        full_text: required
            Individual chat line. Used to infer the context i.e. previous chat history.
        relative_timestamp: optional (broken)
            Used to limit the context to those within `prev_x_seconds`.
            `prev_x_seconds` will be silently ignored if this column doesn't exist.

        Returns:
        ----------
        pd.DataFrame
            The transformed DataFrame
        """

        data = df.copy()
        # Removing empty full text if any
        data = data[data["full_text"].notna()]

        # Built Context
        res_data = []
        max_player_count = 0
        for _, group_df in tqdm(data.groupby(self.context_group_by)):
            # We convert profileID to be from 0 -> however many
            if self.add_player_id:
                group_df["profileid"], _ = group_df["profileid"].factorize()
                max_player_count = max(
                    group_df["profileid"].max() + 1, max_player_count
                )

            for i, df in enumerate(self.differentiate_chat(group_df, self.chat_mode)):
                accumulated_context = self.build_context_for_one_match(
                    df=df.reset_index(drop=True),
                    timestamp_col=self.timestamp_col,
                    past_x_seconds=self.past_x_seconds,
                    context_columns_to_accumulate=self.context_accumulating_columns,
                )

                for accum_col, new_col in zip(
                    self.context_accumulating_columns, self.context_new_column_names
                ):
                    df[new_col] = accumulated_context[accum_col]

                # Workaround:
                # For global, we need the global chat in the context, but the global chat line itself doesn't have to be part of that team.
                if self.chat_mode == "Global":
                    respective_team = group_df["team"].unique()[i]
                    df = df[df["team"] == respective_team].copy()

                res_data.append(df)

        if self.add_player_id:
            logger.debug(f"Max player count: {max_player_count}")

        final_df = pd.concat(res_data)

        if not self.complex_context:
            for col in self.context_new_column_names:
                final_df[col] = final_df[col].apply(lambda row: self.combine_list(row))
        else:
            # We are adding in front of each chat line
            if self.method_to_add_custom_context == "left":
                final_df["context"] = final_df.apply(
                    lambda row: self.collate_context_for_line(row), axis=1
                )

            # we are adding it as speaker segmentation
            if self.method_to_add_custom_context == "below":
                # Context columns stays the same
                self.collate_speaker_segmentation_per_line(final_df)

        return final_df

    @staticmethod
    def differentiate_chat(df: pd.DataFrame, chat_mode: str):
        """
        Given a match in the form of a dataframe, differentiate into four main types of chat:

        Needed columns:
        ---------------
        Global -> uses `is_team_channel` == False (all chat)
               -> uses `team` to differentiate which team the chat belongs to
        Team -> uses `team`
        Personal -> uses `profileid`

        Returns:
        -----------
        1. `Full` -> return df
        2. `Global` -> return list of df, length corresponds to the number of teams that texted.
            Each df is (all chat + team chat + own chat)
        3. `Team` -> return list of df, length corresponds to the number of teams that texted.
            Each df is (team chat + own chat)
        4. `Personal` -> return list of df, length corresponds to the unique number of players in the game that texted
            Each df is own chat.

        """
        if chat_mode == "Full":
            return [df]

        if chat_mode == "Global":
            rez = []

            # Get all chat
            all_chat_filter = df["is_team_channel"] == "false"

            # Unique teams
            teams = df["team"].unique()

            for team in teams:
                team_chat_filter = df["team"] == team
                rez.append(df[all_chat_filter | team_chat_filter].copy())
            return rez

        if chat_mode == "Team":
            rez = []
            for _, group_df in df.groupby("team"):
                rez.append(group_df)
            return rez

        if chat_mode == "Personal":
            rez = []
            for _, group_df in df.groupby("profileid"):
                rez.append(group_df)
            return rez

        raise Exception(
            f"Passed in chat mode of `{chat_mode}`. Must be one of [`Full`, `Global`, `Team`, `Personal`]"
        )

    def collate_context_for_line(self, row: List[List]):
        """
        Collates the context for a single line based on the accumulated.

        Our line should start with:

        [chat type] [team ] [player]

        """
        custom_contexts = []

        # Determine chat type
        if self.add_chat_type:
            chat_types = []
            for is_team_channel in row["is_team_channels"]:
                chat_type = "TEAM_CHAT" if is_team_channel else "ALL_CHAT"
                chat_types.append(chat_type)
            custom_contexts.append(chat_types)

        # Determine team
        if self.add_team_type:
            current_team_id = row["team"]
            team_types = []
            for team in row["teams"]:
                team_type = "ENEMY"
                if team == current_team_id:
                    team_type = "FRIENDLY"
                team_types.append(team_type)
            custom_contexts.append(team_types)

        # Determine player (two teams only)
        if self.add_player_id:
            player_ids = []
            for profile_id in row["profileids"]:
                player_ids.append(f"P{profile_id}")
            custom_contexts.append(player_ids)

        # Create the chat lines
        final_context = []
        for i, context in enumerate(row["context"]):
            for custom_context_info in custom_contexts:
                final_context.append(custom_context_info[i])
            final_context += context
            if self.add_new_line_token:
                final_context.append("NEWLINE")
        return final_context

    def __expand_context_chat_type_for_speaker_segmentation(self, row: List[List]):
        """ """
        result = []
        for is_team_channel, context in zip(row["is_team_channels"], row["context"]):
            val = 2 if is_team_channel else 1
            result += [val] * len(context)
        return result

    def __expand_chat_type_for_speaker_segmentation(self, row: List[List]):
        """ """
        val = 2 if row["is_team_channel"] else 1
        return [val] * len(row["full_text"])

    def __expand_context_team_ids_for_speaker_segmentation(self, row: List[List]):
        """ """
        result = []
        for team, context in zip(row["teams"], row["context"]):
            result += [team + 1] * len(context)
        return result

    def __expand_team_id_for_speaker_segmentation(self, row: List[List]):
        """ """
        return [row["team"] + 1] * len(row["full_text"])

    def __expand_context_player_id_speaker_segmentation(self, row: List[List]):
        """ """
        result = []
        for player_id, context in zip(row["profileids"], row["context"]):
            result += [player_id + 1] * len(context)
        return result

    def __expand_player_id_speaker_segmentation(self, row: List[List]):
        """ """
        return [row["profileid"] + 1] * len(row["full_text"])

    def collate_speaker_segmentation_per_line(self, final_df):
        """
        Collates the dataframe for speaker segmentation. Method is "below".
        WILL EDIT ORIGINAL DF
        """

        # CHAT_TYPE
        # Setup column called CHAT_TYPE -> corresponds to full_text chat_type
        # ALL_CHAT -> 1
        # TEAM_CHAT -> 2
        if self.add_chat_type:
            # Each token in full_text is friendly
            final_df["chat_type"] = final_df.apply(
                lambda row: self.__expand_chat_type_for_speaker_segmentation(row),
                axis=1,
            )
            final_df["context_chat_types"] = final_df.apply(
                lambda row: self.__expand_context_chat_type_for_speaker_segmentation(
                    row
                ),
                axis=1,
            )

        # TEAM_ID
        # T1, T2
        if self.add_team_type:
            final_df["team_id"] = final_df.apply(
                lambda row: self.__expand_team_id_for_speaker_segmentation(row), axis=1
            )
            final_df["context_team_ids"] = final_df.apply(
                lambda row: self.__expand_context_team_ids_for_speaker_segmentation(
                    row
                ),
                axis=1,
            )

        # PLAYER_ID
        # P1 - P10
        if self.add_player_id:
            final_df["player_id"] = final_df.apply(
                lambda row: self.__expand_player_id_speaker_segmentation(row), axis=1
            )
            final_df["context_player_ids"] = final_df.apply(
                lambda row: self.__expand_context_player_id_speaker_segmentation(row),
                axis=1,
            )

        # No new line in this mode as of yet.
        final_df["context"] = final_df["context"].apply(
            lambda row: self.combine_list(row)
        )

    @staticmethod
    def combine_list(row: List[List], middle_tokens: List = None):
        """
        Helper to combine lists:

        Parameter:
        -----------
        row: list[list]

        Returns:
        --------
        list (collapses the outer list)
        """
        res = []
        for x in row:
            res += x
            if middle_tokens is not None:
                res = res + middle_tokens + x
        return res

    @staticmethod
    def build_context_for_one_match(
        df: pd.DataFrame,
        timestamp_col: str,
        past_x_seconds: float,
        context_columns_to_accumulate: List[str],
    ) -> Dict[str, List]:
        """
        Builds the context for one specific match.
        Assumes we have a timestamp column (datetime)

        Parameter:
        ----------
        df: pd.DataFrame
            The dataFrame containing one single match data.
        past_x_seconds: float
            Number of seconds to consider as part of context.
        context_columns_to_accumulate: List[str]
            Name of the columns to accumulate.

        Returns:
        -------
        dict:  {str: List}
            Dictionary with keys as the context_columns_to_accumulate and each entry in the list pertains to the context for each row.
        """
        # Initialize the resulting context
        result = {}
        for col in context_columns_to_accumulate:
            result[col] = [[]]

        # Deal with empty df (should not occur)
        if len(df) == 0:
            return result

        # Create the accumulator for the context
        past_accumulator = {}
        for col in context_columns_to_accumulate:
            past_accumulator[col] = [df.at[0, col]]

        # Include everything if no timestamp.
        if timestamp_col not in df:
            for row_index in range(1, len(df)):
                for col in context_columns_to_accumulate:
                    result[col].append(past_accumulator[col].copy())
                    past_accumulator[col].append(df.at[row_index, col])

            return result

        # We need a pointer to know where are prev timestamp is within the `past_x_seconds`
        prev_pointer = 0

        # Convert column to timestamp (if not already)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # We only need to loop through df once.
        for row_index in range(1, len(df)):
            curr_ts = df.at[row_index, timestamp_col]
            prev_ts = df.at[prev_pointer, timestamp_col]

            # We check if the prev_ts is within past_x_seconds.
            # We move our prev_pointer until we are within past_x_seconds
            while (curr_ts - prev_ts).total_seconds() > past_x_seconds:
                prev_pointer += 1
                prev_ts = df.at[prev_pointer, timestamp_col]

                # We add the context stuff at the end, so we pop to remove the earliest entries.
                for col in context_columns_to_accumulate:
                    past_accumulator[col].pop(0)

            for col in context_columns_to_accumulate:
                result[col].append(past_accumulator[col].copy())
                past_accumulator[col].append(df.at[row_index, col])

        return result


class MergeLinesWithMultipleToxicSpans(DataTransformation):
    """
    Aggregate chat lines with multiple toxic spans
    """

    def __init__(self):
        """
        Initializes merging lines with multiple toxic spans.
        """
        super().__init__()

    def __str__(self):
        return "MergeLinesWithMultipleToxicSpans"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge chat lines with multiple toxic spans.

        Parameter:
        ----------
        df: pd.DataFrame
            The input DataFrame we are going to transform.

        DataFrame Columns:
        -----------------------
        matchid: required
            Unique identifier to group chat into their respective matches.
        line_index: required
            Index of the match.
        start_string_index: required
            Will be converted to a list
        end_string_index: required
            Will be converted to a list
        category_id: required
            Will be converted to a list

        Returns:
        ----------
        pd.DataFrame
            The transformed DataFrame
        """
        data = df.copy()

        # Use the first occurrence for all columns but start_string_index, end_string_index and category_id
        columns = list(data.columns)
        list_columns = ["start_string_index", "end_string_index", "category_id"]
        agg_dict = {}
        for col in columns:
            if col in list_columns:
                agg_dict[col] = list
            else:
                agg_dict[col] = "first"

        data = data.groupby(by=["matchid", "line_index"], as_index=False).agg(agg_dict)

        if "category_id" in data:
            data["min_category_id"] = data["category_id"].apply(lambda x: min(x))
        return data


class ConvertFullTextToWordToken(DataTransformation):
    """
    Tokenizes `full_text` and provides the correct token labels.

    Uses `full_text`, `start_string_index`, `end_string_index` and `category_id`
    """

    def __init__(
        self,
        tokenize_label: bool = True,
        label_strategy: str = "LabelTokensAsIs",
    ):
        """
        Initializes the token label data transformation.
        """
        super().__init__()
        self.tokenize_label = tokenize_label
        try:
            label_func = getattr(LabelStrategy, label_strategy)
            self.tokenize = label_func()
        except:
            logger.error(f"Could not find label strategy: {label_strategy}")

    def __str__(self):
        return "ConvertFullTextToWordToken"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the label for each word where the category_id is used for words in between the start_string_index and end_string_index

        Parameter:
        ----------
        df: pd.DataFrame
            The input DataFrame we are going to transform.

        DataFrame Columns:
        -----------------------
        full_text: required
            Individual chat line. Used to infer the context i.e. previous chat history.
        category_id: required
            Category of the selected span of words.
        start_string_index: required
            Start of the selected span of words
        end_string_index: required
            End of the selected span of words.

        Returns:
        ----------
        pd.DataFrame
            The transformed DataFrame
        """
        data = df.copy()
        # Removing empty full text if any
        data = data[data["full_text"].notna()]

        # Tokenization Strategy
        if self.tokenize_label:
            data["category_id"] = data["category_id"].fillna(0)
            data["label"] = data.apply(
                lambda row: self.tokenize(
                    row["full_text"],
                    row["start_string_index"],
                    row["end_string_index"],
                    row["category_id"],
                ),
                axis=1,
            )

        # Split full_text
        data["full_text"] = data["full_text"].apply(lambda x: x.split())

        return data


class CreateEmptyContext(DataTransformation):
    """ """

    def __init__(self):
        """
        Initializes the token label data transformation.
        """
        super().__init__()

    def __str__(self):
        return "CreateEmptyContext"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the label for each word where the category_id is used for words in between the start_string_index and end_string_index

        Parameter:
        ----------
        df: pd.DataFrame
            The input DataFrame we are going to transform.

        DataFrame Columns:
        -----------------------
        full_text: required
            Individual chat line. Used to infer the context i.e. previous chat history.

        Returns:
        ----------
        pd.DataFrame
            The transformed DataFrame
        """
        data = df.copy()
        # Removing empty full text if any
        data = data[data["full_text"].notna()]

        # Split full_text
        data["context"] = data["full_text"].apply(lambda x: [])
        return data


class AddInfoToCurrentChatLine(DataTransformation):
    """
    Adds appropriate info to the current chat line.
    [CHAT TYPE] [TEAM] [PLAYER_ID]
    """

    def __init__(
        self,
        add_chat_type: bool = False,
        add_team_type: bool = False,
        add_player_id: bool = False,
        update_label: bool = True,
    ):
        super().__init__()
        self.add_chat_type = add_chat_type
        self.add_team_type = add_team_type
        self.add_player_id = add_player_id
        self.total_new_info = sum([add_chat_type, add_team_type, add_player_id])
        self.update_label = update_label

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the appropriate info to full_text and label
        """
        if self.total_new_info == 0:
            return df

        data = df.copy()

        # Update all the labels
        if self.update_label:
            new_labels = [-100] * self.total_new_info
            data["label"] = data["label"].apply(lambda x: new_labels + x)

        # Add info to current_chat_line
        data["full_text"] = data.apply(
            lambda row: self.add_info_to_current_chat_line(row), axis=1
        )
        return data

    def add_info_to_current_chat_line(self, row):
        new_info = []
        if self.add_chat_type:
            chat_type = "TEAM_CHAT" if row["is_team_channel"] else "ALL_CHAT"
            new_info.append(chat_type)

        if self.add_team_type:
            new_info.append("FRIENDLY")

        if self.add_player_id:
            new_info.append(f"P{row['profileid']}")

        return new_info + row["full_text"]


class MapLabels(DataTransformation):
    """
    Given a mapping of labels, convert the existing labels to the new map.
    """

    def __init__(self, map: dict, default_val: int, label_column: str):
        """
        Initializes the word tokenization for the context
        """
        super().__init__()
        self.map = map
        self.label_column = label_column
        self.default_val = default_val

    def __str__(self):
        return f"MapLabels on {self.label_column} with {self.map}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the word tokens for the provided context

        Parameter:
        ----------
        df: pd.DataFrame
            The input DataFrame we are going to transform.

        Returns:
        ----------
        pd.DataFrame
            The transformed DataFrame
        """

        def map_label(labels):
            res = []
            for label in labels:
                if label in self.map:
                    res.append(self.map[label])
                else:
                    res.append(self.default_val)
            return res

        data = df.copy()
        data[self.label_column] = data[self.label_column].apply(lambda x: map_label(x))
        return data
