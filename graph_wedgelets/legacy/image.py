import numpy as np
from PIL import Image

import numpy.typing as npt
from typing import Any
from collections.abc import Iterable


class GridNodes:
    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            if index >= self.width * self.height:
                raise IndexError(
                    f"Index {index} out of bounds (there are only {self.width * self.height} pixels)."
                )
            if index < 0:
                raise IndexError(
                    "Please provide non-negative a non-negative integer as grid index."
                )
            return list(divmod(index, self.width))[::-1]
        elif isinstance(index, slice):
            if (index.step is not None) and (index.step <= 0):
                raise ValueError("Backwards slicing is not supported.")
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
        yield from (
            list(divmod(index, self.width))[::-1]
            for index in range(self.width * self.height)
        )

    def __repr__(self):
        return f"GridNodes(width={self.width}, height={self.height})"


class BinaryWedgeParitioningTree:
    def __init__(
        self,
        nodes: GridNodes,
        signal: npt.NDArray,
        n_blocks: int | Iterable[int] | None,
    ) -> None:
        self.nodes = nodes

        n_blocks_horizontal: int
        n_blocks_vertical: int
        if n_blocks is None:
            n_blocks_horizontal = n_blocks_vertical = 1
        elif isinstance(n_blocks, int):
            n_blocks_horizontal = n_blocks_vertical = n_blocks
        elif isinstance(n_blocks, Iterable):
            n_blocks_horizontal, n_blocks_vertical = n_blocks
        else:
            raise TypeError(
                "Incorrect format for 'n_blocks'. Only None, integers and couples are supported."
            )
        self.partition_size: int = n_blocks_horizontal * n_blocks_vertical

        block_horizontal_size: int = int(
            np.ceil(self.nodes.width / n_blocks_horizontal)
        )
        block_vertical_size: int = int(np.ceil(self.nodes.height / n_blocks_vertical))

        centers: npt.NDArray[np.int64] = (
            np.minimum(
                np.array(
                    [
                        np.tile(
                            np.floor(block_horizontal_size / 2)
                            + block_horizontal_size * np.arange(n_blocks_horizontal),
                            (n_blocks_vertical, 1),
                        ),
                        np.tile(
                            np.floor(block_vertical_size / 2)
                            + block_vertical_size * np.arange(n_blocks_vertical),
                            (n_blocks_horizontal, 1),
                        ).T,
                    ]
                ),
                np.array(
                    [
                        np.full(
                            (n_blocks_vertical, n_blocks_horizontal),
                            self.nodes.width - 1,
                        ),
                        np.full(
                            (n_blocks_vertical, n_blocks_horizontal),
                            self.nodes.height - 1,
                        ),
                    ]
                ),
            )
            .T.reshape((-1, 2))
            .astype(np.int64)
        )

        self.center_nodes: list[int] = (
            (centers @ np.array([1, self.nodes.width], dtype=np.int64))
            .flatten()
            .tolist()
        )
        self.center_nodes.sort()

        self.partition: list[list[int]] = [
            [
                center + offest_hor + self.nodes.width * offset_ver
                for offset_ver in range(
                    -int(np.floor(block_vertical_size / 2)),
                    int(np.floor(block_vertical_size / 2)) + 1,
                )
                for offest_hor in range(
                    -int(np.floor(block_horizontal_size / 2)),
                    int(np.floor(block_horizontal_size / 2)) + 1,
                )
                if 0
                <= center + offest_hor + self.nodes.width * offset_ver
                < self.nodes.width * self.nodes.height
            ]
            for center in self.center_nodes
        ]

        self.mean_signal: npt.NDArray = np.array(
            [signal[block].mean(axis=0) for block in self.partition]
        )

        self.wavelet_coefficients: list[npt.NDArray] = [
            np.vstack((block, np.zeros_like(block))) for block in self.mean_signal
        ]

        self.block_sizes: list[int] = [len(x) for x in self.partition]

        self.error: list[np.floating[Any]] = [
            np.linalg.norm(
                signal[block] - np.tile(self.mean_signal[i], (len(block), 1))
            )
            ** 2
            / (self.nodes.height * self.nodes.width)
            / signal.shape[1]
            for i, block in enumerate(self.partition)
        ]


def to_signal(image: Image.Image) -> tuple[GridNodes, npt.NDArray, int, int]:
    img_array: np.ndarray = np.asarray(image)
    dims: int = np.ndim(img_array)

    if dims == 2:
        return (
            GridNodes(img_array.shape[1], img_array.shape[0]),
            img_array.flatten(order="F"),
            img_array.shape[1],
            img_array.shape[0],
        )
    elif dims == 3:
        indices: np.ndarray = np.indices(
            (img_array.shape[1], img_array.shape[0])
        ).T.reshape(-1, 2)
        return (
            GridNodes(img_array.shape[1], img_array.shape[0]),
            img_array[indices.T[1], indices.T[0], :],
            img_array.shape[1],
            img_array.shape[0],
        )
    raise ValueError("Unsupported image format.")


def from_signal(signal: npt.NDArray, width: int, height: int) -> npt.NDArray:
    if signal.ndim == 1:
        return signal.reshape((height, width))
    elif signal.ndim == 2:
        return signal.reshape((height, width, signal.shape[1]))
    raise ValueError(
        f"The signal has an unforseen number of dimensions ({signal.ndim} ∉ {{1, 2}})."
    )


def _main(test: int) -> None:
    match test:
        case 0:
            with Image.open("tests/easy.bmp") as im:
                arr = np.asarray(im)
            print(arr)
            Image.fromarray(arr).show()
        case 1:
            with Image.open("tests/easy.bmp") as im:
                nodes, signal, w, h = to_signal(im)
            Image.fromarray(from_signal(signal, w, h)).show()
        case 2:
            nodes = GridNodes(5, 4)
            print(nodes)
            print(nodes[2:10:2])
        case 3:
            nodes = GridNodes(5, 4)
            for x in nodes:
                print(x)
        case 4:
            with Image.open("tests/easy.bmp") as im:
                nodes, signal, _, _ = to_signal(im)
            BinaryWedgeParitioningTree(nodes, signal, (4, 3))


if __name__ == "__main__":
    _main(4)
