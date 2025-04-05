"""
Root Mean Square Error (RMSE) Implementation with Multiple Distance Metrics.

This module has an abstract base class for calculating Root Mean Square
Error using different distance metrics. It  both Euclidean and Taxicab
distances. It has incremental updates through operator overloading (+=) and
maintains a running collection of predicted and expected values.

Classes:
    RMSE: Abstract base class for RMSE calculations
    Euclidean: RMSE implementation using Euclidean distance
    Taxicab: RMSE implementation using Manhattan distance
"""

from abc import ABC, abstractmethod
import math


class RMSE(ABC):
    """
    Abstract base class for Root Mean Square Error calculations.

    This class provides a framework for calculating RMSE using different
    distance metrics. Subclasses must implement the distance() method to
    define how differences between predicted and expected values are calculated.

    Attributes:
        _predicted_values (list): Storage for predicted values
        _expected_values (list): Storage for expected values

    Methods:
        __add__: Adds a new prediction-expected pair
        __iadd__: Implements the += operator
        reset: Clears all stored values
        error: Calculates current RMSE (property)
        distance: Abstract method for calculating distance (must be implemented
        by subclasses)
    """

    def __init__(self):
        """Initialize a new RMSE calculator.

        Initializes empty lists for storing predicted and expected values by
        calling the reset() method.
        """
        self.reset()

    def __add__(self, other):
        """Add a new prediction-expected pair to the calculator.

        Args:
            other (tuple): A tuple of length 2 containing:
                - First element: predicted values (tuple/list)
                - Second element: expected values (tuple/list)

        Returns:
            RMSE: Returns self for method chaining

        Raises:
            ValueError: If input is not a tuple of length 2
        """
        if not isinstance(other, tuple) or len(other) != 2:
            raise ValueError("Input must be a tuple of length 2")
        self._predicted_values.append(other[0])
        self._expected_values.append(other[1])
        return self

    def __iadd__(self, other):
        """Implement the += operator for adding prediction-expected pairs.

        This method allows using the += operator as a shorthand for adding
        new predictions, equivalent to calling __add__().

        Args:
            other (tuple): A tuple of length 2 containing predicted and expected
            values

        Returns:
            RMSE: Returns self after adding the new values
        """
        return self.__add__(other)

    def reset(self):
        """Reset the calculator by clearing all stored values.

        Clears both the predicted and expected value lists, effectively
        resetting the error calculation to its initial state.
        """
        self._predicted_values = []
        self._expected_values = []

    @property
    def error(self):
        """Calculate and return the current Root Mean Square Error.

        Computes RMSE using the stored predicted and expected values:
        RMSE = sqrt(sum((distance(predicted, expected))^2) / n)
        where n is the number of predictions.
        Returns:
            float: The calculated RMSE value. Returns 0 if no values are stored.
        """
        if not self._predicted_values or not self._expected_values:
            return 0
        total_error = 0
        n = len(self._predicted_values)
        for predicted, expected in zip(self._predicted_values,
                                       self._expected_values):
            total_error += self.distance(predicted, expected) ** 2
        rmse = math.sqrt(total_error / n)
        return rmse

    @staticmethod
    @abstractmethod
    def distance(predicted, expected):
        """
        Calculate the distance between predicted and expected values.

        Args:
            predicted: Predicted values (typically a tuple or list)
            expected: Expected values (typically a tuple or list)

        Returns:
            float: Distance between the predicted and expected values
        """
        pass


class Euclidean(RMSE):
    """
    RMSE implementation using Euclidean distance.

    This class calculates RMSE using the Euclidean (L2) distance metric, which
    is the square root of the sum of squared differences between corresponding
    elements.

    The Euclidean distance is appropriate when the magnitude of errors in all
    dimensions should be weighted quadratically.
    """

    @staticmethod
    def distance(predicted, expected):
        """
        Calculate Euclidean distance between two points.

        Args:
            predicted (tuple/list): Point in n-dimensional space (predicted
            values)
            expected (tuple/list): Point in n-dimensional space (expected
            values)

        Returns:
            float: Euclidean distance between the points
        """
        return math.sqrt(sum((p - e) ** 2 for p, e in zip(predicted, expected)))


class Taxicab(RMSE):
    """
    RMSE implementation using Taxicab (Manhattan) distance.

    This class calculates RMSE using the Taxicab (L1) distance metric, which is
    the sum of the absolute differences between corresponding elements.

    The Taxicab distance is appropriate when the magnitude of errors in all
    dimensions should be weighted linearly and independently.
    """

    @staticmethod
    def distance(predicted, expected):
        """
        Calculate Taxicab (Manhattan) distance between two points.

        Args:
            predicted (tuple/list): Point in n-dimensional space (predicted
            values)
            expected (tuple/list): Point in n-dimensional space (expected
            values)

        Returns:
            float: Taxicab distance between the points
        """
        return sum(abs(p - e) for p, e in zip(predicted, expected))
