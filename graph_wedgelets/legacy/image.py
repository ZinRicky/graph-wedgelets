import numpy as np
from PIL import Image

# from tqdm import tqdm
# from time import perf_counter

import numpy.typing as npt
from typing import Any, Literal
from collections.abc import Iterable


class NodesGrid:
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
        return np.array([self[x] for x in indices])


class BinaryWedgeParitioningTree:
    def __init__(
        self,
        nodes: NodesGrid,
        signal: npt.NDArray,
        n_blocks: int | Iterable[int] | None,
    ) -> None:
        self.nodes: NodesGrid = nodes
        self.signal: npt.NDArray = signal

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
            [self.signal[block].mean(axis=0) for block in self.partition]
        )

        self.wavelet_coefficients: list[npt.NDArray] = [
            np.vstack((block, np.zeros_like(block))) for block in self.mean_signal
        ]

        self.block_sizes: list[list[int]] = [[len(x), 0] for x in self.partition]

        self.error: list[np.floating] = [
            np.linalg.norm(
                self.signal[block] - np.tile(self.mean_signal[i], (len(block), 1))
            )
            ** 2
            / (self.nodes.height * self.nodes.width)
            / self.signal.shape[1]
            for i, block in enumerate(self.partition)
        ]

    def wedgelet_encode(
        self,
        method: Literal["RA", "MD", "KC", "RA"] | None = None,
        method_parameter: int | None = None,
        tolerance: float = 1e-3,
        metric: float | None = None,
        max_partition_size: int = 1000,
    ):
        max_error_index: np.integer = np.argmax(self.error)
        max_error: np.floating = self.error[max_error_index]

        signal_size: int = self.signal.shape[0]
        new_center_nodes: list[np.integer] = []

        while max_error > tolerance and (
            max_partition_size is None or self.partition_size < max_partition_size
        ):
            new_center_nodes.append(max_error_index)
            current_mean_signal: npt.NDArray = self.mean_signal[max_error_index]
            current_partition: list[int] = self.partition[max_error_index]
            center_index: int = current_partition.index(
                self.center_nodes[max_error_index]
            )

            if method is None:
                method = "RA"

            S: int
            signal_range: npt.NDArray

            match method:
                case "FA":
                    S = len(current_partition)
                    signal_range = np.arange(S)
                case "MD":
                    S = 1
                    signal_range = np.argmax(
                        center_distance(
                            self.nodes.coordinates_from_indices(current_partition),
                            center_index,
                            metric,
                        ),
                        axis=0,
                    )
                case "KC":
                    if method_parameter is None:
                        method_parameter = 2
                    S = min([method_parameter, len(current_partition)])
                    _, signal_range = j_centers(
                        self.nodes.coordinates_from_indices(current_partition),
                        method_parameter,
                        metric=metric,
                    )
                case "RA":
                    rng = np.random.default_rng(seed=1)
                    if method_parameter is None:
                        method_parameter = 2
                    S = min([method_parameter, len(current_partition)])
                    signal_range = rng.integers(len(current_partition), size=S)

            for i in tqdm(range(S)):
                cluster_1: list[int]
                cluster_2: list[int]

                cluster_1, cluster_2 = wedge_split(
                    self.nodes, current_partition, signal_range[i], center_index, metric
                )
                error_1: np.floating = (
                    (
                        np.linalg.norm(
                            self.signal[cluster_1]
                            - np.tile(
                                np.mean(self.signal[cluster_1]), (len(cluster_1), 1)
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
                                np.mean(self.signal[cluster_2]), (len(cluster_2), 1)
                            )
                        )
                        ** 2
                    )
                    / self.signal.shape[0]
                    / self.signal.shape[1]
                )
                error_new: np.floating = error_1 + error_2

                if error_new <= max_error:
                    self.center_nodes[max_error_index] = current_partition[center_index]
                    self.center_nodes.append(current_partition[signal_range[i]])

                    self.partition[max_error_index] = cluster_1
                    self.partition.append(cluster_2)
                    self.mean_signal[max_error_index] = np.mean(
                        self.signal[cluster_1], axis=0
                    )
                    self.mean_signal = np.vstack(
                        (self.mean_signal, np.mean(self.signal[cluster_2], axis=0))
                    )
                    self.wavelet_coefficients.append(
                        np.vstack(
                            (
                                self.mean_signal[max_error_index] - current_mean_signal,
                                self.mean_signal[-1] - current_mean_signal,
                            )
                        )
                    )
                    self.block_sizes.append([len(cluster_1), len(cluster_2)])
                    self.error[max_error_index] = error_1
                    self.error.append(error_2)
                    max_error = error_new

            max_error_index = np.argmax(self.error)
            max_error = self.error[max_error_index]

            self.partition_size += 1


def center_distance(
    coords: npt.NDArray,
    center_index: int,
    metric: float | None = None,
) -> npt.NDArray:
    return np.linalg.norm(
        coords - np.tile(coords[center_index], (len(coords), 1)),
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
    node_coords: npt.NDArray = np.array([node for node in nodes])
    c_1: npt.NDArray[np.integer] = np.argmin(
        np.vstack(
            (
                center_distance(node_coords, center_1, metric),
                center_distance(node_coords, center_2, metric),
            )
        ),
        axis=0,
    )
    return np.nonzero(c_1)[0].tolist(), np.nonzero(1 - c_1)[0].tolist()


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
            BinaryWedgeParitioningTree(nodes, signal, (4, 3))
        case 5:
            print(
                center_distance(
                    np.array([[2, 1], [5, 1], [10, 1], [2, 3], [5, 3], [10, 3]]), 3, 2
                )
            )
        case 6:
            print(wedge_split(NodesGrid(5, 6), [3, 5, 7], 10, 14))
        case 7:
            with Image.open("tests/church.jpg") as im:
                nodes, signal, _, _ = to_signal(im)
            BWP = BinaryWedgeParitioningTree(nodes, signal, (4, 3))
            BWP.wedgelet_encode(max_partition_size=200)


if __name__ == "__main__":
    _main(7)
