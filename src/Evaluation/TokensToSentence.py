from abc import ABC, abstractmethod

import numpy as np

class TokensToSentence(ABC):
    '''
    Convert labels from token level to sentence level.
    '''
    def __init__(self):
        return

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self,
                 token_labels: np.array,
                 first_toxic_label: int = 1,
                 no_toxic_token_found_label: int = 0):
        '''
        Aggregate all the token labels into one.

        Parameters:
        -----------
        token_labels: np.array  (number of samples, max_token_length)
            The token labels to aggregate to a sentence level.
        first_toxic_label: int
            Defaults to 1. Start of the first toxic label.
        no_toxic_token_found_label: int
            Defaults to 0. Label if no toxic label was found.

        Returns:
        --------
        New np.array (number of samples)
        '''
        raise NotImplementedError

class FirstToxicToken(TokensToSentence):
    '''
    Use the first toxic token as the label for the sentence.
    '''
    def __init__(self,
                 first_toxic_label: int = 1,
                 no_toxic_token_found_label: int = 0):
        '''
        Parameters:
        -----------
        first_toxic_label: int
            Defaults to 1. Start of the first toxic label.
        no_toxic_token_found_label: int
            Defaults to 0. Label if no toxic label was found.
        '''
        self.first_toxic_label = first_toxic_label
        self.no_toxic_token_found_label = no_toxic_token_found_label

    def __str__(self):
        return "FirstToxicToken"

    def __call__(self,
                 token_labels: np.array,):
        '''
        Aggregate all the token labels into one by using the first nonzero label.
        We assume the following:
        1. Anything below `first_toxic_label` is non-toxic.

        Parameters:
        -----------
        token_labels: np.array  (number of samples, max_token_length)
            The token labels to aggregate to a sentence level.

        Returns:
        --------
        New np.array (number of samples)
        '''
        new_toxic_labels = []

        for instance in token_labels:

            new_toxic_label = self.no_toxic_token_found_label
            for i in instance:
                if i >= self.first_toxic_label:
                    new_toxic_label = i
                    break

            new_toxic_labels.append(new_toxic_label)

        return np.array(new_toxic_labels)

class MostToxicToken(TokensToSentence):
    '''
    Use the most toxic token as the label for the sentence.
    We assume that toxicity increases as the number decreases.
    i.e.  1 >> 2 > 9
    '''
    def __init__(self,
                 first_toxic_label: int = 1,
                 no_toxic_token_found_label: int = 0):
        '''
        Parameters:
        -----------
        first_toxic_label: int
            Defaults to 1. Start of the first toxic label.
        no_toxic_token_found_label: int
            Defaults to 0. Label if no toxic label was found.
        '''
        self.first_toxic_label = first_toxic_label
        self.no_toxic_token_found_label = no_toxic_token_found_label

    def __str__(self):
        return "MostToxicToken"

    def __call__(self,
                 token_labels: np.array):
        '''
        Aggregate all the token labels into one by choosing the most toxic token found.
        We assume that toxicity increases as the number decreases.
        i.e.  1 >> 2 > 9

        Parameters:
        -----------
        token_labels: np.array  (number of samples, max_token_length)
            The token labels to aggregate to a sentence level.

        Returns:
        --------
        New np.array (number of samples)
        '''
        new_toxic_labels = []

        for instance in token_labels:

            toxic_labels = instance[instance >= self.first_toxic_label]
            new_toxic_label = self.no_toxic_token_found_label

            if (len(toxic_labels) > 0):
                new_toxic_label = np.min(toxic_labels)

            new_toxic_labels.append(new_toxic_label)

        return np.array(new_toxic_labels)

class MostFrequentToxicToken(TokensToSentence):
    '''
    Use the most frequent toxic token as the label for the sentence.
    If there is a tie, it uses the most toxic label.
    '''
    def __init__(self,
                 first_toxic_label: int = 1,
                 no_toxic_token_found_label: int = 0):
        '''
        Parameters:
        -----------
        first_toxic_label: int
            Defaults to 1. Start of the first toxic label.
        no_toxic_token_found_label: int
            Defaults to 0. Label if no toxic label was found.
        '''
        self.first_toxic_label = first_toxic_label
        self.no_toxic_token_found_label = no_toxic_token_found_label

    def __str__(self):
        return "MostFrequentToxicToken"

    def __call__(self,
                 token_labels: np.array
                 ):
        '''
        Aggregate all the token labels into one by choosing the most toxic token found.
        We assume that toxicity increases by the label number.
        i.e. 0 < 1 < 2 << 9

        Parameters:
        -----------
        token_labels: np.array  (number of samples, max_token_length)
            The token labels to aggregate to a sentence level.

        Returns:
        --------
        New np.array (number of samples)
        '''
        new_toxic_labels = []

        for instance in token_labels:

            toxic_labels = instance[instance >= self.first_toxic_label]
            new_toxic_label = self.no_toxic_token_found_label

            if (len(toxic_labels) > 0):
                toxic_labels = toxic_labels.astype(int)
                new_toxic_label = np.bincount(toxic_labels).argmax()

            new_toxic_labels.append(new_toxic_label)

        return np.array(new_toxic_labels)

class BinaryToxicToken(TokensToSentence):
    '''
    Changes the multi-class toxic labels to binary labels of toxic or not.
    0 - Non-toxic
    1 - Toxic
    '''
    def __init__(self,
                 first_toxic_label: int = 1,
                 no_toxic_token_found_label: int = 0):
        '''
        Parameters:
        -----------
        first_toxic_label: int
            Defaults to 1. Start of the first toxic label.
        no_toxic_token_found_label: int
            Defaults to 0. Label if no toxic label was found.
        '''
        self.first_toxic_label = first_toxic_label
        self.no_toxic_token_found_label = no_toxic_token_found_label

    def __str__(self):
        return "BinaryToxicToken"

    def __call__(self,
                 token_labels: np.array):
        '''
        Aggregate all the token labels into one by choosing the binary label of toxic or not.

        Parameters:
        -----------
        token_labels: np.array  (number of samples, max_token_length)
            The token labels to aggregate to a sentence level.
        first_toxic_label: int
            Defaults to 1. Start of the first toxic label.
        no_toxic_token_found_label: int
            Defaults to 0. Label if no toxic label was found.

        Returns:
        --------
        New np.array (number of samples)
        '''
        new_toxic_labels = []

        for instance in token_labels:

            toxic_labels = instance[instance >= self.first_toxic_label]
            new_toxic_label = min(len(toxic_labels), 1)
            new_toxic_labels.append(new_toxic_label)

        return np.array(new_toxic_labels)