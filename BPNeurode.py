"""CS3B Assignment - Back Propagation Neural Network Implementation.

This module implements a back-propagation neural network node (BPNeurode) that
inherits from the base Neurode class. It handles the backward pass of the neural
network training process, calculating deltas and updating weights.

Author: Ethan Jordan
Date: 11/1/2024
"""

from Neurode import Neurode


class BPNeurode(Neurode):
    """A Back Propagation Neural Network Node.

    Implements the backward pass functionality needed for training a neural
    network, including delta calculation and weight updates.
    """

    def __init__(self):
        """Initialize BPNeurode with default values."""
        super().__init__()
        self._delta = 0.0

    @staticmethod
    def _sigmoid_derivative(value: float) -> float:
        """Calculate the derivative of the sigmoid function.

        Args:
            value: Input value for derivative calculation

        Returns:
            float: Calculated sigmoid derivative
        """
        return value * (1 - value)

    def _calculate_delta(self, target_value=None):
        """Calculate the delta value for the node.

        Args:
            target_value: Expected output for output layer nodes (optional)
        """
        if target_value is not None:
            # For output layer nodes
            self._delta = ((target_value - self._value) *
                          self._sigmoid_derivative(self._value))
        else:
            # For hidden layer nodes
            weighted_sum = 0.0
            for node in self._neighbors[self.Side.DOWNSTREAM]:
                weighted_sum += node.delta * node._weights[self]
            self._delta = self._sigmoid_derivative(self._value) * weighted_sum

    @property
    def delta(self):
        """Get the node's delta value.

        Returns:
            float: Current delta value
        """
        return self._delta

    def data_ready_downstream(self, node: Neurode) -> None:
        """Process data ready signal from downstream node.

        Args:
            node: The downstream node signaling readiness
        """
        if self._check_in(node, self.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def _update_weights(self):
        """Update weights for all downstream connections."""
        learning_rate = 0.05
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            # Calculate adjustment for the downstream node
            adjustment = learning_rate * node.delta * self._value
            # Have the downstream node adjust its weights
            node.adjust_weights(self, adjustment)

    def adjust_weights(self, node, adjustment):
        """Adjust weights for a specific node connection.

        Args:
            node: The connected node
            adjustment: Weight adjustment value
        """
        self._weights[node] += adjustment

    def set_expected(self, target_value: float) -> None:
        """Set expected output value and start backpropagation.

        Args:
            target_value: Expected output value
        """
        self._calculate_delta(target_value)
        self._fire_upstream()

    def _fire_upstream(self) -> None:
        """Signal all upstream nodes that data is ready."""
        for node in self._neighbors[self.Side.UPSTREAM]:
            node.data_ready_downstream(self)
