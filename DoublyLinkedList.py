"""
This module implements a Doubly Linked List data structure.

It provides classes for creating and manipulating a doubly linked list,
which is a linked list that is traversable both forwards and backwards.
"""


class DLLNode:
    """
    A class that is the building blocks of the linked list.

    Each node has data and links/memory addresses to the next and previous
    nodes.
    """

    def __init__(self, data):
        """
        Initialize a new node.

        Args:
            data: The data to be stored in the node.
        """
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """A class creating a doubly linked list.

    This class implements a linked list that is traversable both forwards
    and backwards. It provides methods for adding and removing elements,
    traversing the list, and performing basic operations.
    """

    def __init__(self):
        """Initialize an empty doubly linked list."""
        self._head = None
        self._tail = None
        self._curr = None

    def add_to_head(self, data):
        """
        Add a new node with the given data to the head of the list.

        Args:
            data: The data to be added to the list.
        """
        new_node = DLLNode(data)
        if not self._head:
            self._head = self._tail = new_node
        else:
            new_node.next = self._head
            self._head.prev = new_node
            self._head = new_node
        self._curr = self._head

    def add_after_current(self, data):
        """
        Add a new node with inputted data after the current node.

        If the list is empty, raise an IndexError.

        Args:
            data: The data to be added to the list.
        """
        if self._curr is None:
            raise IndexError("Cannot add after current when list is empty.")
        new_node = DLLNode(data)
        new_node.prev = self._curr
        new_node.next = self._curr.next
        if self._curr.next:
            self._curr.next.prev = new_node
        else:
            self._tail = new_node
        self._curr.next = new_node

    def remove_from_head(self):
        """Remove the node at the head of the list."""
        if not self._head:
            raise IndexError("Cannot remove from head of an empty list.")
        old_head = self._head
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self._curr = self._head
        return old_head.data

    def remove_after_current(self):
        """Remove the node after the current node."""
        if self._curr is None or self._curr.next is None:
            raise IndexError("No node to remove after current.")
        to_remove = self._curr.next
        self._curr.next = to_remove.next
        if to_remove.next:
            to_remove.next.prev = self._curr
        else:
            self._tail = self._curr
        return to_remove.data

    def reset_to_head(self):
        """Set the current node to the head of the list."""
        self._curr = self._head

    def reset_to_tail(self):
        """Set the current node to the tail of the list."""
        self._curr = self._tail

    def move_forward(self):
        """Move the current node one step forward in the list."""
        if self._curr is None or self._curr.next is None:
            raise IndexError("Cannot move forward; at the end of the list.")
        self._curr = self._curr.next

    def move_backward(self):
        """Move the current node one step backward in the list."""
        if self._curr is None or self._curr.prev is None:
            raise IndexError("Cannot move backward; at the start of the list.")
        self._curr = self._curr.prev

    def find(self, data):
        """
        Find a node with the given data in the list.

        Args:
            data: The data to search for.

        Returns:
            The data of the found node.

        Raises:
            IndexError: If the data is not found.
        """
        current = self._head
        while current:
            if current.data == data:
                self._curr = current
                return data
            current = current.next
        raise IndexError("Data not found in the list.")

    def remove(self, data):
        """
        Remove the first occurrence of a node with the given data.

        Args:
            data: The data to be removed.

        Raises:
            IndexError: If the data is not found.
        """
        current = self._head
        while current:
            if current.data == data:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self._head = current.next
                if current.next:
                    current.next.prev = current.prev
                else:
                    self._tail = current.prev
                if self._curr == current:
                    self._curr = self._head
                return data
            current = current.next
        raise IndexError("Data not found to remove.")

    @property
    def curr_data(self):
        """
        Get the data of the current node.

        Returns:
            The data of the current node.

        Raises:
            IndexError: If the current node is None.
        """
        if self._curr is None:
            raise IndexError("Current node is None.")
        return self._curr.data

    def is_empty(self):
        """
        Check if the list is empty.

        Returns:
            bool: True if the list is empty, False otherwise.
        """
        return self._head is None

    def print_list(self):
        """Print all elements in the list."""
        curr_temp = self._curr
        self.reset_to_head()
        while self._curr:
            print(self._curr.data)
            self._curr = self._curr.next
        self._curr = curr_temp
