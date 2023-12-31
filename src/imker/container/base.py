from collections import OrderedDict
from keyword import iskeyword
from typing import Any, TypeVar

# Refer to https://github.com/cdgriffith/Box

T = TypeVar("T")


class DataContainer(OrderedDict[str, T]):
    """
    A dict-like class that provides dot access to its elements.
    """

    def __getattr__(self, key: str) -> Any:
        """
        Get the value of the given key using dot notation.

        Args:
            key (str): The key to access.

        Returns:
            Any: The value associated with the key.

        Raises:
            AttributeError: If the key is not present in the dictionary.
        """
        if key in self:
            return self[key]
        raise AttributeError(f"'Container' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set the value of the given key using dot notation.

        Args:
            key (str): The key to set.
            value (Any): The value to associate with the key.
        """
        self[key] = value

    def __dir__(self) -> list[Any]:
        items = set(super().__dir__())
        for key in self.keys():
            key = str(key)
            if key.isidentifier() and not iskeyword(key):
                items.add(key)

        return list(items)
