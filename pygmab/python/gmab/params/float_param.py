from functools import cached_property

from gmab.params.base_param import BaseParam


class FloatParam(BaseParam):
    """
    A class representing a float parameter.
    """

    def __init__(self, low: float, high: float, size: int = 1, step: float = 0.1):
        """
        Creates a FloatParam that will suggest float values during the optimization.

        The parameter can either be a float, or a list of floats, depending on the specified
        size. The values sampled by the optimizaton will be limited to the specified granularity,
        lower and upper bounds.

        Args:
            low (float): The lower bound of the suggested values.
            high (float)): The upper bound of the suggested values.
            size (int): The size if the parameter shall be a list of integers. Default is 1.
            step (float): The step size between the suggested values. Default is 1.0.

        Returns:
            FloatParam: An instance of the parameter with the specified properties.

        Raises:
            ValueError: If low is not an float, if high is not an float that is greater than
            low, or if size is not a positive integer, or if step is not a positive float.

        Example:
        >>> param = FloatParam(low=1.0, high=10.0, size=3, step=0.1)
        >>> print(param)
        ... FloatParam(low=1.0, high=10.0, size=3, step=0.1)
        """
        if high <= low:
            raise ValueError("high must be a float that is greater than low.")
        if step <= 0:
            raise ValueError("step must be positive float.")

        super().__init__(size)
        self.low: float = float(low)
        self.high: float = float(high)
        self.step: float = float(step)

    def __repr__(self):
        return f"FloatParam(low={self.low}, high={self.high}, size={self.size}, step={self.step})"

    @cached_property
    def bounds(self) -> list[tuple]:
        """
        Calculate and return the parameter's internal bounds for the optimization.

        The bounds will be used as constraints for the internal representation (or actions)
        of the optimization algorithm about the parameter's value

        Returns:
            list[tuple]: A list of tuples representing the bounds
        """
        if self.step == 1.0:
            upper_bound = self.high
        else:
            upper_bound = (self.high - self.low) // self.step
            upper_bound += 1 if (self.high - self.low) % self.step != 0 else 0
            upper_bound += int(self.low)
        return [(int(self.low), int(upper_bound))] * self.size

    def map_to_value(self, actions: list[int]) -> float | list[float]:
        """
        Maps an action by the optimization problem to the value of the parameter.

        Args:
            actions (list[int]): A list of integer to map.

        Returns:
            int | list[int]: The resulting integer value(s).
        """
        actions = [float(min(self.low + x * self.step, self.high)) for x in actions]

        if self.size == 1:
            return actions[0]
        return actions
