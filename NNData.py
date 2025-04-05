"""Module for managing neural network data."""

from enum import Enum
import numpy as np
from collections import deque
import random


class Order(Enum):
    """Enumeration for data ordering methods."""

    SHUFFLE = 0
    STATIC = 1


class Set(Enum):
    """Enumeration for dataset types."""

    TRAIN = 0
    TEST = 1


class NNData:
    """Class for managing neural network data."""

    @staticmethod
    def percentage_limiter(percentage: float) -> float:
        """
        Limit a percentage value to the range [0, 1].

        Args: The percentage value to be limited.
        Returns: The limited percentage value.
        """
        return min(1, max(0, percentage))

    def __init__(self, features=None, labels=None, train_factor=0.9):
        """
        Initialize the neural network data manager.

        Args features: The input features.
        Args labels: The corresponding labels.
        Args train_factor: The proportion of data to use for training.
        """
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        self._features = None
        self._labels = None
        self.load_data(features, labels)

    def load_data(self, features=None, labels=None):
        """
        Load and validate the dataset.

        Args features: The features of the dataset.
        Args labels: The labels of the dataset.
        :raises ValueError: If features and labels are incompatible.
        """
        if features is None or labels is None:
            self._features = None
            self._labels = None
        elif len(features) != len(labels):
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError("Features and labels must have the same length")
        else:
            try:
                self._features = np.array(features, dtype=float)
                self._labels = np.array(labels, dtype=float)
            except ValueError:
                self._features = None
                self._labels = None
                self.split_set()
                raise ValueError(
                    "Features and labels must be convertible to float"
                )
        self.split_set()

    def split_set(self, new_train_factor=None):
        """
        Split the dataset into training and testing sets.

        Args new_train_factor: Optional new proportion of data to be
        used for training.
        """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        if self._features is None or self._labels is None:
            self._train_indices = []
            self._test_indices = []
            return
        num_samples = len(self._features)
        num_train = int(num_samples * self._train_factor)
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)
        self._train_indices = all_indices[:num_train]
        self._test_indices = all_indices[num_train:]

    def prime_data(self, target_set=None, order=Order.STATIC):
        """
        Prepare the data pools for training or testing.

        Args target_set: The dataset to prime (TRAIN, TEST, or both).
        Args order: The order in which to access data (SHUFFLE, STATIC).
        """
        if target_set in {Set.TRAIN, None}:
            self._train_pool = deque(self._train_indices)
            if order == Order.SHUFFLE:
                train_list = list(self._train_pool)
                random.shuffle(train_list)
                self._train_pool = deque(train_list)

        if target_set in {Set.TEST, None}:
            self._test_pool = deque(self._test_indices)
            if order == Order.SHUFFLE:
                test_list = list(self._test_pool)
                random.shuffle(test_list)
                self._test_pool = deque(test_list)

    def get_one_item(self, target_set=None):
        """
        Retrieve one item from the specified data pool.

        Args target_set: The dataset to retrieve from (TRAIN or TEST).
        Returns: A tuple containing a feature and its corresponding
        label, or None.
        """
        if self._features is None or self._labels is None:
            return None

        if target_set == Set.TEST:
            pool = self._test_pool
        else:
            pool = self._train_pool

        if not pool:
            return None

        index = pool.popleft()
        return self._features[index], self._labels[index]

    def number_of_samples(self, target_set=None):
        """
        Get the number of samples in the specified dataset.

        Args target_set: The dataset to count (TRAIN or TEST).
        Returns: The number of samples in the dataset.
        """
        if target_set == Set.TEST:
            return len(self._test_indices)
        if target_set == Set.TRAIN:
            return len(self._train_indices)
        return len(self._train_indices) + len(self._test_indices)

    def pool_is_empty(self, target_set=None):
        """
        Check if the specified data pool is empty.

        Args target_set: The data pool to check (TRAIN or TEST).
        Returns: True if the data pool is empty, False otherwise.
        """
        if target_set == Set.TEST:
            return len(self._test_pool) == 0
        return len(self._train_pool) == 0
