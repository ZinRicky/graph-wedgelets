import numpy as np
from PIL import Image

from tqdm import tqdm
import json
import time

import numpy.typing as npt
from typing import Any, Literal
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
        yield from (
            list(divmod(index, self.width))[::-1]
            for index in range(self.width * self.height)
        )

    def __repr__(self):
        return f"NodesGrid(width={self.width}, height={self.height})"

    def coordinates_from_indices(self, indices: list[int]) -> npt.NDArray:
        return self.asarray()[indices]

    def asarray(self) -> npt.NDArray:
        if self.array is None:
            x: npt.NDArray[np.integer]
            y: npt.NDArray[np.integer]
            x, y = np.meshgrid(range(self.width), range(self.height))
            return np.column_stack((x.ravel(), y.ravel()))
        return self.array


class BinaryWedgeParitioningTree:
    def __init__(
        self,
        nodes: NodesGrid,
        signal: npt.NDArray,
        n_blocks: int | Iterable[int] | None,
        max_partition_size: int,
    ) -> None:
        self.nodes: NodesGrid = nodes
        self.signal: npt.NDArray = signal
        self.max_partition_size: int = max_partition_size

        n_blocks_horizontal: int
        n_blocks_vertical: int
        if n_blocks is None:
            n_blocks_horizontal = n_blocks_vertical = 1
        elif isinstance(n_blocks, int):
            n_blocks_horizontal = n_blocks_vertical = n_blocks
        elif isinstance(n_blocks, Iterable) and len(list(n_blocks)) == 2:
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

        self.partition: dict[int, list[int]] = {
            i: [
                center + horizontal_offest + self.nodes.width * vertical_offset
                for vertical_offset in range(
                    -int(np.floor(block_vertical_size / 2)),
                    int(np.floor(block_vertical_size / 2)) + 1,
                )
                for horizontal_offest in range(
                    -int(np.floor(block_horizontal_size / 2)),
                    int(np.floor(block_horizontal_size / 2)) + 1,
                )
                if 0
                <= center + horizontal_offest + self.nodes.width * vertical_offset
                < self.nodes.width * self.nodes.height
            ]
            for i, center in enumerate(self.center_nodes)
        }

        self.mean_signal: npt.NDArray = np.zeros(
            (self.max_partition_size, self.signal.shape[1])
        )
        for i, block in self.partition.items():
            self.mean_signal[i] = self.signal[block].mean(axis=0)

        self.wavelet_coefficients: npt.NDArray = np.zeros(
            (self.max_partition_size, self.signal.shape[1], 2)
        )
        for i, block in self.partition.items():
            self.wavelet_coefficients[i, :, 0] = self.mean_signal[i]

        self.block_sizes: dict[int, list[int]] = {
            key: [len(block), 0] for key, block in self.partition.items()
        }

        self.error: dict[int, np.floating] = {
            i: np.linalg.norm(
                self.signal[block] - np.tile(self.mean_signal[i], (len(block), 1))
            )
            ** 2
            / (self.nodes.height * self.nodes.width)
            / self.signal.shape[1]
            for i, block in self.partition.items()
        }

    def wedgelet_encode(
        self,
        method: Literal["RA", "MD", "KC", "RA"] | None = None,
        method_parameter: int | None = None,
        tolerance: float = 1e-3,
        metric: float | None = None,
    ) -> None:
        max_error: np.floating = np.max(list(self.error.values()))
        max_error_index: int = list(self.error.keys())[
            list(self.error.values()).index(max_error)
        ]

        signal_size: int = self.signal.shape[0]

        while max_error > tolerance and (
            self.max_partition_size is None
            or self.partition_size < self.max_partition_size
        ):
            center_node: int = self.center_nodes[max_error_index]
            current_mean_signal: npt.NDArray = self.mean_signal[max_error_index]
            current_partition: list[int] = self.partition[max_error_index]
            center_index: int = current_partition.index(center_node)

            if method is None:
                method = "RA"

            S: int
            new_node_range: npt.NDArray

            match method:
                case "FA":
                    S = len(current_partition)
                    new_node_range = np.arange(S)
                case "MD":
                    S = 1
                    new_node_range = np.argmax(
                        center_distance(
                            self.nodes.coordinates_from_indices(current_partition),
                            center_index,
                            metric,
                        ),
                        axis=0,
                        keepdims=True,
                    )
                case "KC":
                    raise NotImplementedError()
                    # if method_parameter is None:
                    #     method_parameter = 2
                    # S = min([method_parameter, len(current_partition)])
                    # _, signal_range = j_centers(
                    #     self.nodes.coordinates_from_indices(current_partition),
                    #     method_parameter,
                    #     metric=metric,
                    # )
                case "RA":
                    rng = np.random.default_rng(seed=1)
                    if method_parameter is None:
                        method_parameter = 2
                    current_partition_size: int = len(current_partition)
                    S = (
                        method_parameter
                        if method_parameter < current_partition_size
                        else current_partition_size
                    )
                    new_node_range = rng.choice(
                        current_partition,
                        size=S,
                        replace=False,
                    )

            for possible_node in new_node_range:
                cluster_1: list[int]
                cluster_2: list[int]

                cluster_1, cluster_2 = wedge_split(
                    self.nodes,
                    current_partition,
                    int(possible_node),
                    center_node,
                    metric,
                )

                t_0 = time.perf_counter()
                error_1: np.floating = (
                    (
                        np.linalg.norm(
                            self.signal[cluster_1]
                            - np.tile(
                                np.mean(self.signal[cluster_1], axis=0),
                                (len(cluster_1), 1),
                            )
                        )
                        ** 2
                    )
                    / self.signal.shape[0]
                    / self.signal.shape[1]
                )
                error_2: np.floating = (
                    (
                        np.linalg.norm(
                            self.signal[cluster_2]
                            - np.tile(
                                np.mean(self.signal[cluster_2], axis=0),
                                (len(cluster_2), 1),
                            )
                        )
                        ** 2
                    )
                    / self.signal.shape[0]
                    / self.signal.shape[1]
                )
                error_new: np.floating = error_1 + error_2
                t_1 = time.perf_counter()

                if error_new <= max_error:
                    t_0 = time.perf_counter()
                    if len(self.center_nodes) == self.partition_size + 1:
                        self.center_nodes[-1] = int(possible_node)
                    else:
                        self.center_nodes.append(int(possible_node))

                    self.partition[max_error_index] = cluster_1

                    self.partition[self.partition_size] = cluster_2

                    self.mean_signal[max_error_index] = np.mean(
                        self.signal[cluster_1], axis=0
                    )
                    self.mean_signal[self.partition_size] = np.mean(
                        self.signal[cluster_2], axis=0
                    )

                    self.wavelet_coefficients[self.partition_size, :, 0] = (
                        self.mean_signal[max_error_index] - current_mean_signal
                    )
                    self.wavelet_coefficients[self.partition_size, :, 1] = (
                        self.mean_signal[self.partition_size] - current_mean_signal
                    )

                    self.block_sizes[self.partition_size] = [
                        len(cluster_1),
                        len(cluster_2),
                    ]
                    t_1 = time.perf_counter()

                    self.error[max_error_index] = error_1
                    self.error[self.partition_size] = error_2
                    max_error = error_new

            t_0 = time.perf_counter()
            max_error = np.max(list(self.error.values()))
            max_error_index = list(self.error.keys())[
                list(self.error.values()).index(max_error)
            ]
            t_1 = time.perf_counter()

            self.partition_size += 1

        if self.partition_size < self.max_partition_size:
            self.mean_signal = self.mean_signal[: self.partition_size]
            self.wavelet_coefficients = self.wavelet_coefficients[: self.partition_size]

    def wedgelet_decode(
        self, metric: float | None = None
    ) -> tuple[npt.NDArray, dict[int, list[int]]]:
        new_signal: npt.NDArray = np.zeros_like(self.signal)
        for i, center_node in enumerate(self.center_nodes):
            new_signal[self.partition[i]] = self.mean_signal[i]
        return (
            new_signal,
            {node: self.partition[i] for i, node in enumerate(self.center_nodes)},
        )


def center_distance(
    coords: npt.NDArray,
    center_index: int,
    metric: float | None = None,
) -> npt.NDArray:
    return np.linalg.norm(
        coords - np.tile(coords[center_index], (coords.shape[0], 1)),
        ord=metric,
        axis=1,
    )


def j_centers(
    coords: npt.NDArray,
    n_clusters: int,
    greedy_search_subset: list[int] | None = None,
    input_centers: list[int] | int | None = None,
    metric: float | None = None,
) -> tuple[list[int], npt.NDArray]:
    N: int = coords.shape[0]

    input_center: list[int]
    if input_centers is None:
        rng = np.random.default_rng()
        input_center = [rng.integers(N)]
    elif isinstance(input_centers, int):
        input_center = [input_centers]
    else:
        input_center = input_centers.copy()

    if greedy_search_subset is None:
        greedy_search_subset = list(range(N))

    distances: npt.NDArray = np.zeros((n_clusters, N))
    centers: list[int] = [-1 for _ in range(n_clusters)]
    centers[: len(input_center)] = input_center

    for j in range(n_clusters):
        distances[j] = center_distance(coords, centers[j], metric)
        if j >= len(input_center) and j != n_clusters - 1:
            centers[j + 1] = greedy_search_subset[
                np.argmax(np.min(distances[:j, greedy_search_subset], axis=0))
            ]

    return centers, distances.min(axis=0)


def wedge_split(
    nodes: NodesGrid,
    indices: list[int],
    center_1: int,
    center_2: int,
    metric: float | None = None,
) -> tuple[list[int], list[int]]:
    node_coords: npt.NDArray = nodes.asarray()
    c_1: npt.NDArray[np.integer] = np.argmin(
        np.vstack(
            (
                center_distance(node_coords, center_1, metric),
                center_distance(node_coords, center_2, metric),
            )
        ),
        axis=0,
    )
    c_2: npt.NDArray[np.integer] = np.zeros_like(c_1)
    c_2[indices] = np.ones(len(indices))
    return np.nonzero(c_1 * c_2)[0].tolist(), np.nonzero((1 - c_1) * c_2)[0].tolist()


def to_signal(image: Image.Image) -> tuple[NodesGrid, npt.NDArray, int, int]:
    return grid_to_signal(np.asarray(image))


def grid_to_signal(img_array: npt.NDArray) -> tuple[NodesGrid, npt.NDArray, int, int]:
    dims: int = np.ndim(img_array)

    if dims == 2:
        return (
            NodesGrid(img_array.shape[1], img_array.shape[0]),
            img_array.flatten(order="F"),
            img_array.shape[1],
            img_array.shape[0],
        )
    elif dims == 3:
        indices: np.ndarray = np.indices(
            (img_array.shape[1], img_array.shape[0])
        ).T.reshape(-1, 2)
        return (
            NodesGrid(img_array.shape[1], img_array.shape[0]),
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
            nodes = NodesGrid(5, 4)
            print(nodes)
            print(nodes[2:10:2])
        case 3:
            nodes = NodesGrid(5, 4)
            for x in nodes:
                print(x)
        case 4:
            with Image.open("tests/easy.bmp") as im:
                nodes, signal, _, _ = to_signal(im)
            BinaryWedgeParitioningTree(nodes, signal, (4, 3), 10)
        case 5:
            print(
                center_distance(
                    np.array([[2, 1], [5, 1], [10, 1], [2, 3], [5, 3], [10, 3]]), 3, 2
                )
            )
        case 6:
            print(wedge_split(NodesGrid(4, 4), [0, 1, 2, 3, 5, 6, 12, 13], 5, 6))
        case 7:
            with Image.open("tests/easy.bmp") as im:
                nodes, signal, width, height = to_signal(im)
            BWP = BinaryWedgeParitioningTree(nodes, signal, (2, 2), 20)
            BWP.wedgelet_encode()
            # Image.fromarray(from_signal(BWP.mean_signal, 4, 3), mode="RGB").show()
        case 8:
            with Image.open("tests/church.jpg") as im:
                nodes, signal, width, height = to_signal(im)

            print("Starting signal translation")
            t_0 = time.perf_counter()
            BWP = BinaryWedgeParitioningTree(nodes, signal, 1, 50)
            t_1 = time.perf_counter()
            print(f"Initialization time: {t_1-t_0:.3e} s.\n")

            print("Starting encoding")
            t_0 = time.perf_counter()
            BWP.wedgelet_encode(method_parameter=16)
            t_1 = time.perf_counter()
            print(f"Encoding time: {t_1-t_0:.3e} s.\n")

            print("Starting decoding")
            t_0 = time.perf_counter()
            s, P = BWP.wedgelet_decode()
            t_1 = time.perf_counter()
            print(f"Decoding time: {t_1-t_0:.3e} s.\n")

            Image.fromarray(from_signal(s, width, height), mode="RGB").show()
            np.savez_compressed(
                "test.npz", s=BWP.mean_signal, P=np.array(list(P.keys())), metric=2
            )


if __name__ == "__main__":
    _main(8)
