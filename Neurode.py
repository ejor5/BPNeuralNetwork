#!/usr/bin/env python3
"""
Neurode Implementation Module.

This module implements the base classes for a Feed-forward backpropagation
neural network. It contains MultiLinkNode as an abstract base class and
Neurode as its concrete implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
import random


class MultiLinkNode(ABC):
    """
    Abstract base class for neural network nodes.

    This class provides the foundation for implementing neural network nodes
    with upstream and downstream connections. It manages node relationships
    and reporting states using binary encoding.
    """

    class Side(Enum):
        """
        Enumeration for specifying node relationship directions.

        Attributes:
            UPSTREAM: Represents input connections
            DOWNSTREAM: Represents output connections
        """

        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self) -> None:
        """
        Initialize a new MultiLinkNode.

        Creates dictionaries to track reporting status, reference values,
        and neighboring nodes for both upstream and downstream connections.
        """
        self._reporting_nodes = {self.Side.UPSTREAM: 0, self.Side.DOWNSTREAM: 0}
        self._reference_value = {self.Side.UPSTREAM: 0, self.Side.DOWNSTREAM: 0}
        self._neighbors = {self.Side.UPSTREAM: [], self.Side.DOWNSTREAM: []}

    def __str__(self) -> str:
        """
        Return a string representation of the node and its connections.

        Returns:
            str: Formatted string showing node ID and neighbor IDs
        """
        return (
            f"Node ID: {id(self)}\n"
            f"Upstream neighbors: "
            f"{[id(node) for node in self._neighbors[self.Side.UPSTREAM]]}\n"
            f"Downstream neighbors: "
            f"{[id(node) for node in self._neighbors[self.Side.DOWNSTREAM]]}"
        )

    @abstractmethod
    def _process_new_neighbor(self, node: MultiLinkNode, side: Side) -> None:
        """
        Process a newly added neighboring node.

        Args:
            node: The newly added neighboring node
            side: Indicates whether the neighbor is upstream or downstream

        Note:
            This is an abstract method that must be implemented by subclasses.
        """
        pass

    def reset_neighbors(self, nodes: list, side: Side) -> None:
        """
        Reset and initialize the neighboring nodes for a given side.

        Args:
            nodes: List of nodes to set as neighbors
            side: Indicates whether to set upstream or downstream neighbors

        Note:
            This method also calculates the reference value for checking
            when all nodes have reported.
        """
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = 2 ** len(nodes) - 1


class Neurode(MultiLinkNode):
    """
    Implementation of a neural network node.

    This class extends MultiLinkNode to provide specific functionality
    for neural network operations including weight management and
    learning rate controls.

    Attributes:
        _learning_rate (float): Class-wide learning rate for all neurodes
    """

    _learning_rate = 0.05

    @property
    def learning_rate(self) -> float:
        """
        Get the current learning rate.

        Returns:
            float: The current learning rate
        """
        return self.__class__._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate for all neurodes.

        Args:
            learning_rate: New learning rate value
        """
        self.__class__._learning_rate = learning_rate

    def __init__(self) -> None:
        """
        Initialize a new Neurode.

        Creates a neurode with initial value of 0 and empty weights dictionary.
        """
        super().__init__()
        self._value = 0
        self._weights = {}

    def _process_new_neighbor(
            self, node: Neurode, side: MultiLinkNode.Side) -> None:
        """
        Process a newly added neighboring node.

        For upstream neighbors, assigns a random initial weight.

        Args:
            node: The newly added neighboring node
            side: Indicates whether the neighbor is upstream or downstream
        """
        if side == self.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node: Neurode, side: MultiLinkNode.Side) -> bool:
        """
        Register that a neighboring node has information available.

        Uses binary encoding to track which nodes have reported in.

        Args:
            node: The reporting neighboring node
            side: Indicates whether the neighbor is upstream or downstream

        Returns:
            bool: True if all nodes on the given side have reported, False
            otherwise
        """
        node_index = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= 1 << node_index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        return False

    def get_weight(self, node: Neurode) -> float:
        """
        Get the weight associated with an upstream node.

        Args:
            node: The upstream node to get the weight for

        Returns:
            float: The weight value for the given node
        """
        return self._weights.get(node, 0)

    @property
    def value(self) -> float:
        """
        Get the current value of the neurode.

        Returns:
            float: The current value
        """
        return self._value
