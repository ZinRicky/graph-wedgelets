# Necessary libraries
import numpy as np
from PIL import Image
from copy import deepcopy

# Typing libraries
import numpy.typing as npt
from typing import Literal, TypedDict
from collections.abc import Iterable


class NodesGrid:
    def __init__(self, width: int, height: int) -> None:
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
            return np.array(divmod(index, self.width)[::-1])
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
            return self.asarray()[start:stop:step]
        else:
            raise TypeError("Incorrect index specification.")

    def __iter__(self):
        yield from self.asarray().__iter__()

    def __repr__(self):
        return f"<NodesGrid(width={self.width}, height={self.height}, memoized_array={self.array is not None})>"

    def asarray(self) -> npt.NDArray:
        if self.array is None:
            x: npt.NDArray[np.integer]
            y: npt.NDArray[np.integer]
            x, y = np.meshgrid(range(self.width), range(self.height))
            self.array = np.column_stack((x.flatten(), y.flatten()))
        return self.array

    def coordinates_from_indices(self, indices: npt.ArrayLike) -> npt.NDArray:
        return self.asarray()[np.asarray(indices)]


class BWPExport(TypedDict):
    center_nodes: npt.NDArray
    next_nodes: npt.NDArray
    grid_width: int
    grid_height: int
    block_horizontal_size: int
    block_vertical_size: int
    initial_partition_size: int
    mean_signal: npt.NDArray
    metric: float


class BinaryWedgePartitioningTree:
    def __init__(
        self,
        nodes: NodesGrid,
        signal: npt.NDArray,
        n_blocks: int | Iterable[int] | None,
        max_partition_size: int,
    ) -> None:
        self.nodes: NodesGrid = nodes
        self.max_partition_size: int = max_partition_size

        if signal.ndim not in (1, 2):
            raise ValueError(
                f"The provided signal has an unsupported number of dimensions ({signal.ndim} ∉ {{1, 2}})."
            )

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

        self.block_horizontal_size: int = int(
            np.ceil(self.nodes.width / n_blocks_horizontal)
        )
        self.block_vertical_size: int = int(
            np.ceil(self.nodes.height / n_blocks_vertical)
        )

        centers: npt.NDArray[np.integer] = grid_centers(
            self.nodes,
            self.block_horizontal_size,
            self.block_vertical_size,
            n_blocks_horizontal,
            n_blocks_vertical,
        )
        self.center_nodes: list[int] = (
            (centers @ np.array([1, self.nodes.width], dtype=np.int64))
            .flatten()
            .tolist()
        )
        self.center_nodes.sort()
        self.initial_center_nodes: list[int] = self.center_nodes.copy()
        self.initial_partition_size: int = self.partition_size
        self.next_nodes: list[int] = []

        self.partition: dict[int, npt.NDArray[np.integer]] = grid_partition(
            self.center_nodes,
            self.nodes.width,
            self.nodes.height,
            self.block_horizontal_size,
            self.block_vertical_size,
        )
        self.initial_partition: dict[int, npt.NDArray[np.integer]] = (
            self.partition.copy()
        )

        self.mean_signal: npt.NDArray
        if signal.ndim == 1:
            self.mean_signal = np.zeros(self.max_partition_size)
            for i, block in self.partition.items():
                self.mean_signal[i] = signal[block].mean()
        else:
            self.mean_signal = np.zeros((self.max_partition_size, signal.shape[1]))
            for i, block in self.partition.items():
                self.mean_signal[i] = signal[block].mean(axis=0)

        self.wavelet_coefficients: npt.NDArray
        if signal.ndim == 1:
            self.wavelet_coefficients = np.zeros((self.max_partition_size, 1, 2))
        else:
            self.wavelet_coefficients = np.zeros(
                (self.max_partition_size, signal.shape[1], 2)
            )
        for i, block in self.partition.items():
            self.wavelet_coefficients[i, :, 0] = self.mean_signal[i]

        self.block_sizes: dict[int, list[int]] = {
            key: [len(block), 0] for key, block in self.partition.items()
        }

        coeff: float = 1 / self.nodes.height / self.nodes.width
        if signal.ndim == 2:
            coeff /= signal.shape[1]

        self.error: npt.NDArray = np.zeros(self.max_partition_size)
        if signal.ndim == 1:
            self.error[: self.initial_partition_size] = [
                np.linalg.norm(signal[block] - self.mean_signal[i]) ** 2 / coeff
                for i, block in sorted(self.partition.items())
            ]
        else:
            self.error[: self.initial_partition_size] = [
                np.linalg.norm(
                    signal[block] - np.tile(self.mean_signal[i], (len(block), 1))
                )
                ** 2
                / coeff
                for i, block in sorted(self.partition.items())
            ]

    def wedgelet_encode(
        self,
        signal: npt.NDArray,
        method: Literal["FA", "MD", "KC", "RA"] | None = None,
        method_parameter: int | None = None,
        tolerance: float = 1e-3,
        max_partition_size: int | None = None,
        metric: float | None = None,
    ) -> None:
        self.metric: float = 2 if metric is None else metric
        max_error_index: int = int(np.argmax(self.error))
        max_error: np.floating = self.error[max_error_index]

        while max_error > tolerance and self.partition_size < np.min(
            [
                self.max_partition_size,
                max_partition_size if max_partition_size is not None else np.inf,
            ]
        ):
            self.next_nodes.append(max_error_index)
            center_node: int = self.center_nodes[max_error_index]
            current_mean_signal: npt.NDArray = self.mean_signal[max_error_index]
            current_partition: npt.NDArray[np.integer] = self.partition[max_error_index]
            center_index: int = int(np.where(current_partition == center_node)[0][0])

            if method is None:
                method = "RA"

            S: int
            new_node_range: npt.NDArray

            match method:
                case "FA":
                    S = len(current_partition)
                    new_node_range = current_partition.copy()
                case "MD":
                    S = 1
                    new_node_range = current_partition[
                        np.argmax(
                            center_distance(
                                self.nodes.coordinates_from_indices(current_partition),
                                center_index,
                                metric,
                            ),
                            axis=0,
                            keepdims=True,
                        )
                    ]
                case "KC":
                    if method_parameter is None:
                        method_parameter = 2
                    S = min([method_parameter, len(current_partition) - 1])
                    node_range: list[int] = j_centers(
                        self.nodes.coordinates_from_indices(current_partition),
                        method_parameter,
                        input_centers=center_index,
                        metric=metric,
                    )[0]
                    new_node_range = current_partition[node_range]
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
                        current_partition, size=S, replace=False
                    )

            for idx, possible_node in enumerate(new_node_range):
                cluster_1: npt.NDArray[np.integer]
                cluster_2: npt.NDArray[np.integer]

                cluster_1, cluster_2 = wedge_split(
                    self.nodes,
                    current_partition,
                    int(possible_node),
                    center_node,
                    metric,
                )

                error_1: np.floating = (
                    (
                        np.linalg.norm(
                            signal[cluster_1]
                            - np.tile(
                                np.mean(signal[cluster_1], axis=0),
                                (
                                    (len(cluster_1),)
                                    if signal.ndim == 1
                                    else (len(cluster_1), 1)
                                ),
                            )
                        )
                        ** 2
                    )
                    / signal.shape[0]
                    / (signal.shape[1] if signal.ndim > 1 else 1)
                )
                error_2: np.floating = (
                    (
                        np.linalg.norm(
                            signal[cluster_2]
                            - np.tile(
                                np.mean(signal[cluster_2], axis=0),
                                (
                                    (len(cluster_2),)
                                    if signal.ndim == 1
                                    else (len(cluster_2), 1)
                                ),
                            )
                        )
                        ** 2
                    )
                    / signal.shape[0]
                    / (signal.shape[1] if signal.ndim > 1 else 1)
                )
                error_new: np.floating = error_1 + error_2

                if error_new < max_error:
                    if len(self.center_nodes) == self.partition_size + 1:
                        self.center_nodes[-1] = int(possible_node)
                    else:
                        self.center_nodes.append(int(possible_node))

                    self.partition[max_error_index] = cluster_1

                    self.partition[self.partition_size] = cluster_2

                    self.mean_signal[max_error_index] = np.mean(
                        signal[cluster_1], axis=0
                    )
                    self.mean_signal[self.partition_size] = np.mean(
                        signal[cluster_2], axis=0
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

                    self.error[max_error_index] = error_1
                    self.error[self.partition_size] = error_2
                    max_error = error_new

            max_error_index = int(np.argmax(self.error))
            max_error = self.error[max_error_index]

            self.partition_size += 1

        if max_partition_size is None and self.partition_size < self.max_partition_size:
            self.mean_signal = self.mean_signal[: self.partition_size]
            self.wavelet_coefficients = self.wavelet_coefficients[: self.partition_size]

    def export(self) -> BWPExport:
        new_center_nodes: npt.NDArray = np.asarray(self.center_nodes)
        max_node: np.integer = np.max(new_center_nodes)

        cn_export_type: type = (
            np.uint8
            if max_node < np.iinfo(np.uint8).max
            else (
                np.uint16
                if max_node < np.iinfo(np.uint16).max
                else np.uint32 if max_node < np.iinfo(np.uint32).max else np.uint64
            )
        )

        new_next_nodes: npt.NDArray = np.asarray(self.next_nodes)
        max_next_node: np.integer = np.max(new_next_nodes)
        nn_export_type: type = (
            np.uint8
            if max_next_node < np.iinfo(np.uint8).max
            else (
                np.uint16
                if max_next_node < np.iinfo(np.uint16).max
                else np.uint32 if max_next_node < np.iinfo(np.uint32).max else np.uint64
            )
        )

        return BWPExport(
            {
                "center_nodes": new_center_nodes.astype(cn_export_type),
                "next_nodes": new_next_nodes.astype(nn_export_type),
                "grid_width": self.nodes.width,
                "grid_height": self.nodes.height,
                "block_horizontal_size": self.block_horizontal_size,
                "block_vertical_size": self.block_vertical_size,
                "initial_partition_size": self.initial_partition_size,
                "mean_signal": self.mean_signal.astype(np.uint8),
                "metric": self.metric,
            }
        )

    def save_npz(self, path: str) -> None:
        np.savez_compressed(path, **self.export())

    def save(self, path: str) -> None:
        np.savez(path, **self.export())


class BWPDecoder:
    def __init__(
        self,
        center_nodes: npt.NDArray[np.integer],
        next_nodes: npt.NDArray[np.integer],
        grid_width: int,
        grid_height: int,
        block_horizontal_size: int,
        block_vertical_size: int,
        mean_signal: npt.NDArray,
        initial_partition_size: int,
        metric: float | None = None,
    ) -> None:
        self.nodes: NodesGrid = NodesGrid(grid_width, grid_height)
        self.center_nodes: npt.NDArray[np.integer] = center_nodes.copy()
        self.next_nodes: npt.NDArray[np.integer] = next_nodes.copy()
        self.mean_signal: npt.NDArray = mean_signal.copy()
        self.initial_partition_size: int = initial_partition_size
        self.partition: dict[int, npt.NDArray[np.integer]] = grid_partition(
            self.center_nodes[:initial_partition_size].flatten().tolist(),
            grid_width,
            grid_height,
            block_horizontal_size,
            block_vertical_size,
        )
        self.metric: float | None = metric
        self._is_decoded: bool = False

    def wedgelet_decode(
        self,
    ) -> tuple[npt.NDArray, dict[int, npt.NDArray[np.integer]]]:
        if not self._is_decoded:
            for i, node in enumerate(self.next_nodes):
                current_partition: npt.NDArray[np.integer] = self.partition[int(node)]
                old_node: int = self.center_nodes[node]
                new_node: int = self.center_nodes[i + self.initial_partition_size]

                cluster_1: npt.NDArray
                cluster_2: npt.NDArray
                cluster_1, cluster_2 = wedge_split(
                    self.nodes,
                    current_partition,
                    new_node,
                    old_node,
                    self.metric,
                )

                self.partition[int(node)] = cluster_1
                self.partition[i + self.initial_partition_size] = cluster_2

            signal: npt.NDArray
            if self.mean_signal.ndim == 1:
                signal = np.zeros(self.nodes.width * self.nodes.height)
                for i, block in self.partition.items():
                    signal[block] = self.mean_signal[i]
            else:
                signal = np.zeros(
                    (self.nodes.width * self.nodes.height, self.mean_signal.shape[1])
                )
                for i, block in self.partition.items():
                    signal[block] = np.tile(self.mean_signal[i], (block.shape[0], 1))

            self.signal: npt.NDArray = signal
            self._is_decoded = True

        return self.signal, self.partition


def grid_centers(
    nodes: NodesGrid,
    block_horizontal_size: int,
    block_vertical_size: int,
    n_blocks_horizontal: int,
    n_blocks_vertical: int,
) -> npt.NDArray[np.integer]:
    return (
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
                        nodes.width - 1,
                    ),
                    np.full(
                        (n_blocks_vertical, n_blocks_horizontal),
                        nodes.height - 1,
                    ),
                ]
            ),
        )
        .T.reshape((-1, 2))
        .astype(np.int64)
    )


def grid_partition(
    center_nodes: list[int],
    grid_width: int,
    grid_height: int,
    block_horizontal_size: int,
    block_vertical_size: int,
) -> dict[int, npt.NDArray[np.integer]]:
    return {
        i: np.array(
            [
                center + horizontal_offest + grid_width * vertical_offset
                for vertical_offset in range(
                    -int(np.floor(block_vertical_size / 2)),
                    int(np.floor(block_vertical_size / 2)) + 1,
                )
                for horizontal_offest in range(
                    -int(np.floor(block_horizontal_size / 2)),
                    int(np.floor(block_horizontal_size / 2))
                    + (
                        center % grid_width
                        != -int(np.floor(block_horizontal_size / 2)) % grid_width
                    ),
                )
                if (
                    0
                    <= center + horizontal_offest + grid_width * vertical_offset
                    < grid_width * grid_height
                )
            ]
        )
        for i, center in enumerate(center_nodes)
    }


def center_distance(
    coords: npt.NDArray,
    center_index: int,
    metric: float | None = None,
) -> npt.NDArray:
    vectors: npt.NDArray[np.floating] = coords - coords[center_index]

    if metric is None or metric == 2:
        return np.sqrt(np.sum(np.square(vectors), axis=1))
    elif metric == 1:
        return np.sum(np.abs(vectors), axis=0)
    elif metric == np.inf:
        return np.max(np.abs(vectors), axis=0)
    else:
        return np.linalg.norm(vectors, ord=metric, axis=1)


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
        rng = np.random.default_rng(seed=1)
        input_center = [int(rng.integers(N))]
    elif isinstance(input_centers, int):
        input_center = [input_centers]
    else:
        input_center = input_centers.copy()

    if greedy_search_subset is None:
        greedy_search_subset = list(range(N))

    distances: npt.NDArray = np.zeros((n_clusters, N))
    centers: list[int] = [-1 for _ in range(n_clusters)]
    centers[: len(input_center)] = input_center

    for j in range(n_clusters - 1):
        distances[j] = center_distance(coords, centers[j], metric)
        if j >= len(input_center) - 1:
            centers[j + 1] = greedy_search_subset[
                np.argmax(np.min(distances[: j + 1, greedy_search_subset], axis=0))
            ]
    distances[-1] = center_distance(coords, centers[-1], metric)
    return centers, distances.argmin(axis=0)


def wedge_split(
    nodes: NodesGrid,
    indices: npt.ArrayLike,
    center_1: int,
    center_2: int,
    metric: float | None = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    new_indices: npt.NDArray = np.asarray(indices)
    node_coords: npt.NDArray = nodes.coordinates_from_indices(new_indices)

    new_center_1: int = int(np.nonzero(new_indices == center_1)[0][0])
    new_center_2: int = int(np.nonzero(new_indices == center_2)[0][0])

    c_1: npt.NDArray[np.floating] = np.heaviside(
        center_distance(node_coords, new_center_1, metric)
        - center_distance(node_coords, new_center_2, metric),
        0,
    )

    return new_indices[np.nonzero(c_1)[0]], new_indices[np.nonzero((1 - c_1))[0]]


def to_signal(image: Image.Image) -> tuple[NodesGrid, npt.NDArray, int, int]:
    return grid_to_signal(np.asarray(image))


def grid_to_signal(img_array: npt.NDArray) -> tuple[NodesGrid, npt.NDArray, int, int]:
    dims: int = np.ndim(img_array)

    if dims == 2:
        return (
            NodesGrid(img_array.shape[1], img_array.shape[0]),
            img_array.flatten(),
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
