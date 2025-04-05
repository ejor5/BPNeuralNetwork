"""
Neurode Inheritance Module.

This module implements the base classes from Neurode.py, the base class for a
feed-forward Neural Network.
"""
from __future__ import annotations
from Neurode import Neurode
import numpy as np


class FFNeurode(Neurode):
    """
    FFNeurode class.

    This class inherits from Neurode and implements methods specific to
    feed-forward neural network nodes.
    """

    def __init__(self):
        """Initialize the FFNeurode instance."""
        super().__init__()

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Calculate the sigmoid function of a given value.

        Args:
            value (float): The value to be transformed using the sigmoid
            function.

        Returns:
            float: The result of the sigmoid function.
        """
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self) -> None:
        """
        Calculate the weighted sum of upstream nodes' values.

        Apply the sigmoid function, and store the result in self._value.
        """
        weighted_sum = sum(
            self.get_weight(node) * node.value
            for node in self._neighbors[self.Side.UPSTREAM]
        )
        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self) -> None:
        """Call data_ready_upstream on each downstream neighbor."""
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node: Neurode) -> None:
        """
        Register that an upstream node has data ready.

        If all upstream nodes have data, calculate the value and fire to
        downstream nodes.

        Args:
            node (Neurode): The upstream node reporting data readiness.
        """
        if self._check_in(node, self.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float) -> None:
        """
        Set the value of an input layer neurode and signal downstream neighbors.

        Args:
            input_value (float): The value to set for the input layer
            neurode.
        """
        self._value = input_value
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)
