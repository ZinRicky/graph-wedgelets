# Necessary libraries
import numpy as np
from PIL import Image

# Typing libraries
import numpy.typing as npt
from typing import Literal
from collections.abc import Iterable


class NodesGrid:
    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height
        self.shape: tuple[int, int] = (self.width, self.height)
        self.array: npt.NDArray | None = None

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            if index >= self.width * self.height:
                raise IndexError(
                    f"Index {index} out of bounds (there are only {self.width * self.height} pixels)."
                )
            if index < 0:
                raise IndexError("Please provide a non-negative integer as grid index.")
            return list(divmod(index, self.width))[::-1]
        elif isinstance(index, slice):
            if (index.step is not None) and (index.step <= 0):
                raise ValueError("Backwards slicing is not supported for GridNodes.")
            start: int = 0 if index.start is None else max([0, index.start])
            stop: int = (
                self.width * self.height
                if index.stop is None
                else min([self.width * self.height, index.stop])
            )
            step: int = 1 if index.step is None else index.step
            return [list(divmod(x, self.width))[::-1] for x in range(start, stop, step)]
        else:
            raise TypeError("Incorrect index specification.")

    def __iter__(self):
        yield from self.asarray().__iter__()

    def __repr__(self):
        return f"<NodesGrid(width={self.width}, height={self.height}, memoized_array={self.array is not None})>"

    def coordinates_from_indices(self, indices: list[int] | npt.NDArray) -> npt.NDArray:
        return self.asarray()[indices]

    def asarray(self) -> npt.NDArray:
        if self.array is None:
            x: npt.NDArray[np.integer]
            y: npt.NDArray[np.integer]
            x, y = np.meshgrid(range(self.width), range(self.height))
            self.array = np.column_stack((x.ravel(), y.ravel()))
        return self.array
