# Operative imports
import numpy as np
from PIL import Image
import copy
from tqdm import tqdm

# Type annotations
from numpy.typing import NDArray


class Nodes:
    def __init__(self, nodes):
        if isinstance(nodes, np.ndarray) and np.ndim(nodes) == 2:
            self.nodes = copy.deepcopy(nodes)
        else:
            self.nodes = np.vstack(nodes)


def _distcenter(
    nodes: Nodes,
    center_index: int,
    ord: float | None = None,
) -> NDArray:
    if (ord is not None) and (not isinstance(ord, float) or ord <= 0):
        raise ValueError(
            f"'ord' must be either a positive number or inf; got {ord} instead."
        )
    packed_nodes = nodes.nodes
    center = np.tile(packed_nodes[center_index], (packed_nodes.shape[0], 1))
    return np.linalg.norm(packed_nodes - center, ord=ord, axis=1)


def to_signal(image: Image.Image) -> tuple[Nodes, NDArray, int, int]:
    img_array: np.ndarray = np.asarray(image)
    dims: int = np.ndim(img_array)

    if dims == 2:
        return (
            Nodes(
                np.indices((img_array.shape[1], img_array.shape[0])).T.reshape(-1, 2)
            ),
            img_array.flatten(order="F"),
            img_array.shape[1],
            img_array.shape[0],
        )
    if dims == 3:
        indices: np.ndarray = np.indices(
            (img_array.shape[1], img_array.shape[0])
        ).T.reshape(-1, 2)
        return (
            Nodes(indices),
            img_array[indices.T[1], indices.T[0], :],
            img_array.shape[1],
            img_array.shape[0],
        )
    raise ValueError("Unsupported image format.")


class BinaryWedgeParitioningTree:
    def __init__(
        self,
        nodes: Nodes,
        signal: NDArray,
        width: int,
        height: int,
        n_blocks_hor: int,
        n_blocks_ver: int,
        max_partition_size: int,
    ):
        self.nodes = copy.deepcopy(nodes.nodes)
        self.center_nodes: NDArray = np.zeros((max_partition_size, 2))
        self.signal: NDArray = np.zeros((max_partition_size, signal.shape[1]))
        self.wavelet_coefficients: NDArray = np.zeros(
            (max_partition_size, signal.shape[1], 2)
        )
        self.block_sizes: NDArray = np.zeros((max_partition_size, 2))
        self.error: NDArray = np.zeros(max_partition_size)
        self.partitions: list[NDArray] = []
        self.partition_size = n_blocks_hor * n_blocks_ver

        n_pixels = width * height
        block_width = np.ceil(width / n_blocks_hor)
        block_height = np.ceil(height / n_blocks_ver)
        nodes_x = self.nodes[:, 0]
        nodes_y = self.nodes[:, 1]

        try:
            for i in range(n_blocks_hor):
                for j in range(n_blocks_ver):
                    self.center_nodes[i * n_blocks_hor + j, 0] = np.nonzero(
                        (nodes_x == np.min(np.ceil(block_width / 2) + i * block_width))
                        & (
                            nodes_y
                            == np.min(np.ceil(block_height / 2) + i * block_height)
                        )
                    )[0].tolist()[0]

                    self.partitions.append(
                        np.nonzero(
                            (nodes_x < (i + 1) * block_width)
                            & (nodes_x >= i * block_width)
                            & (nodes_y < (j + 1) * block_height)
                            & (nodes_y >= j * block_height)
                        )[0]
                    )
        except IndexError:
            raise ValueError(
                f"Exceeded maximum partition size of {max_partition_size} during BWP tree initialisation."
            )

        for block in range(self.partition_size):
            self.signal[block] = np.mean(signal[self.partitions[block]], axis=0)
            self.wavelet_coefficients[block, :, 0] = self.signal[block].copy()
            self.block_sizes[block, 0] = self.partitions[block].shape[0]
            self.error[block] = (
                np.linalg.norm(
                    signal[self.partitions[block]]
                    - np.tile(self.signal[block], (self.partitions[block].shape[0], 1)),
                    "fro",
                )
                ** 2
                / n_pixels
                / signal.shape[1]
            )


def _main() -> None:
    with Image.open("tests/test-tr-col.png") as im:
        V, f, dim_x, dim_y = to_signal(im)

    J_x, J_y = 5, 5
    tol = 1e-3
    metric = 2

    BWP = BinaryWedgeParitioningTree(V, f, dim_x, dim_y, J_x, J_y, 1000)

    print(BWP.error)


if __name__ == "__main__":
    _main()
