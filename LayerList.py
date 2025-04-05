"""
LayerList Implementation Module.

This module extends the DoublyLinkedList to create a LayerList
specifically for a feed-forward backpropagation neural network.
It initializes input and output layers with the given number of
neurodes and allows for the addition and removal of hidden layers.
"""
from DoublyLinkedList import DoublyLinkedList


class LayerList(DoublyLinkedList):
    """Layerlist class for the layers of the Network.

    The layerlist class extends DoublyLinkedList and implements
    it specifically to the feed-forward backpropagation neural
    network.

    Args:
        DoublyLinkedList (class): Inheritance of the DLL
    """

    def _link_with_next(self):
        """Connect neurodes in current and next node bidirectionally."""
        for node in self._curr.data:
            node.reset_neighbors(self._curr.next.data,
                                 self._neurode_type.Side.DOWNSTREAM)
        for node in self._curr.next.data:
            node.reset_neighbors(self._curr.data,
                                 self._neurode_type.Side.UPSTREAM)

    def __init__(self, inputs, outputs, neurode_type):
        """
        Initialize the LayerList with input & output neurodes and connect them.

        Args:
            inputs (int): Number of input neurodes.
            outputs (int): Number of output neurodes.
            neurode_type: Type of neurode to be used.
        """
        super().__init__()
        self._neurode_type = neurode_type
        if inputs < 1 or outputs < 1:
            raise ValueError
        input_layer = [neurode_type() for _ in range(inputs)]
        output_layer = [neurode_type() for _ in range(outputs)]
        self.add_to_head(input_layer)
        self.add_after_current(output_layer)
        self._link_with_next()

    def add_layer(self, num_nodes):
        """
        Add a hidden layer after the current layer.

        Args:
            num_nodes: Number of total Nodes.
        """
        if self._curr == self._tail:
            raise IndexError
        if num_nodes < 0:
            raise ValueError
        hidden_layer = [self._neurode_type() for _ in range(num_nodes)]
        self.add_after_current(hidden_layer)
        self._link_with_next()
        self.move_forward()
        self._link_with_next()
        self.move_backward()

    def remove_layer(self):
        """Remove the layer after the current layer."""
        if self._curr == self._tail or self._curr.next == self._tail:
            raise IndexError
        self.remove_after_current()
        self._link_with_next()

    @property
    def input_nodes(self):
        """Return the list of input neurodes."""
        return self._head.data

    @property
    def output_nodes(self):
        """Return the list of output neurodes."""
        return self._tail.data
