import math
from functools import cached_property

from gmab.params.base_param import BaseParam


class FloatParam(BaseParam):
    """
    A class representing a float parameter.
    """

    def __init__(
        self, low: float, high: float, size: int = 1, steps: float = 100, log: bool = False
    ):
        """
        Creates a FloatParam that will suggest float values during the optimization.

        The parameter can either be a float, or a list of floats, depending on the specified
        size. The values sampled by the optimizaton will be limited to the specified granularity,
        lower and upper bounds.

        Args:
            low (float): The lower bound of the suggested values.
            high (float): The upper bound of the suggested values.
            size (int): The size if the parameter shall be a list of floats. Default is 1.
            steps (int): The number of steps between low and high. Default is 100.
            log (bool): A flag to indicate log-transformation. Default is False.

        Returns:
            FloatParam: An instance of the parameter with the specified properties.

        Raises:
            ValueError: If low is not an float, if high is not an float that is greater than
            low, or if size is not a positive integer, or if step is not a positive float.

        Example:
        >>> param = FloatParam(low=1.0, high=10.0, size=3, step=0.1)
        >>> print(param)
        FloatParam(low=1.0, high=10.0, size=3, step=0.1)
        """
        if high <= low:
            raise ValueError("high must be a float that is greater than low.")
        if steps <= 0:
            raise ValueError("steps must be positive integer.")

        super().__init__(size)
        self.log: bool = bool(log)
        self.low: float = float(low)
        self.high: float = float(high)
        self.steps: int = int(steps)

    def __repr__(self):
        repr = f"FloatParam(low={self.low}, high={self.high}, size={self.size}, "
        repr += f"steps={self.steps}, log={self.log})"
        return repr

    @cached_property
    def _offset(self):
        """Only relevant if self.log"""
        return 1 - self.low

    @cached_property
    def _low_trans(self):
        if self.log:
            return math.log(self.low + self._offset)
        return self.low

    @cached_property
    def _step(self):
        high_trans = math.log(self.high + self._offset) if self.log else self.high
        return (high_trans - self._low_trans) / self.steps

    @cached_property
    def bounds(self) -> list[tuple]:
        """
        Calculate and return the parameter's internal bounds for the optimization.

        The bounds will be used as constraints for the internal representation (or actions)
        of the optimization algorithm about the parameter's value

        Returns:
            list[tuple]: A list of tuples representing the bounds
        """
        return [(0, self.steps)] * self.size

    def map_to_value(self, actions: list[int]) -> float | list[float]:
        """
        Maps an action by the optimization problem to the value of the parameter.

        Args:
            actions (list[int]): A list of integer to map.

        Returns:
            float | list[float]: The resulting float value(s).
        """
        # Apply scaling
        actions = [self._low_trans + self._step * x for x in actions]

        # Optional log-transformation
        if self.log:
            actions = [math.exp(x) - self._offset for x in actions]

        if len(actions) == 1:
            return actions[0]
        return actions
