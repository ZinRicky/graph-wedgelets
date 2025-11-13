# Necessary libraries
import numpy as np
import igraph as ig
import random
from copy import deepcopy

# Typing libraries
import numpy.typing as npt
from typing import Literal


class BinaryWedgePartitioningTree:
    def __init__(
        self,
        graph: ig.Graph,
        signal: npt.NDArray,
        max_partition_size: int,
        initial_partition_size: int | None = None,
        initial_partition_method: (
            Literal["betweenness", "closeness", "random"] | None
        ) = None,
        use_leiden: bool = False,
        center_nodes: npt.ArrayLike | None = None,
    ) -> None:
        if signal.ndim not in (1, 2):
            raise ValueError(
                f"The provided signal has an unsupported number of dimensions ({signal.ndim} ∉ {{1, 2}})."
            )
        if center_nodes is not None and np.asarray(center_nodes).ndim != 1:
            raise ValueError(
                f"The specified center nodes should be provided in a 1-dimensional array-like."
            )

        self.graph: ig.Graph = deepcopy(graph)

        if (
            initial_partition_size is not None
            and len(self.graph.vs) < initial_partition_size
        ):
            raise ValueError(
                f"The provided initial partition size {initial_partition_size} is larger than the graph size {len(self.graph.vs)}."
            )

        self.graph.vs["original_id"] = [x.index for x in self.graph.vs]

        self.center_nodes: list[int] = []

        _auto_centers: bool = False

        if initial_partition_size is None and center_nodes is None:
            raise ValueError(
                "At least one between 'initial_partition_size' and 'center_nodes' should not be None."
            )
        elif center_nodes is not None:
            if np.asarray(center_nodes).shape[0] >= max_partition_size:
                self.center_nodes = (
                    np.asarray(center_nodes, dtype=np.int64)
                    .flatten()
                    .tolist()[:max_partition_size]
                )
            else:
                self.center_nodes = (
                    np.asarray(center_nodes, dtype=np.int64).flatten().tolist()
                )
                if (
                    initial_partition_size is not None
                    and len(self.center_nodes) >= initial_partition_size
                ):
                    self.center_nodes = self.center_nodes[:initial_partition_size]
                elif initial_partition_size is not None:
                    _auto_centers = True
        else:
            _auto_centers = True

        if _auto_centers:
            _partition_method: Literal["betweenness", "closeness", "random"]
            if initial_partition_method is None:
                _partition_method = "random"
            elif initial_partition_method in ("betweenness", "closeness", "random"):
                _partition_method = initial_partition_method
            else:
                raise ValueError(
                    "Unrecognized partition method. Supported methods: {'betweenness', 'closeness', 'random'}."
                )

            if use_leiden or initial_partition_size is None:
                random.seed(1)
                clusters: list[ig.Graph] = sorted(
                    self.graph.community_leiden(
                        objective_function="modularity", n_iterations=-1
                    ).subgraphs(),
                    key=lambda g: len(g.vs),
                    reverse=True,
                )

                for cluster in clusters:
                    match _partition_method:
                        case "random":
                            leiden_rng: np.random.Generator = np.random.default_rng(
                                seed=1
                            )
                            cluster.vs["score"] = leiden_rng.choice(
                                len(cluster.vs), len(cluster.vs), replace=False
                            ).tolist()
                        case "betweenness":
                            cluster.vs["score"] = cluster.betweenness()
                        case "closeness":
                            cluster.vs["score"] = cluster.closeness()

                if initial_partition_size is None:
                    for cluster in clusters:
                        next_node = cluster.vs[
                            int(np.flip(np.argsort(cluster.vs["score"]))[0])
                        ]["original_id"]

                        if next_node not in self.center_nodes:
                            self.center_nodes.append(next_node)

                    if len(self.center_nodes) > max_partition_size:
                        self.center_nodes = self.center_nodes[:max_partition_size]
                else:
                    i: int = 0

                    while (
                        len(self.center_nodes) < initial_partition_size
                        or i < initial_partition_size
                    ):
                        for cluster in clusters:
                            try:
                                next_node = cluster.vs[
                                    int(np.flip(np.argsort(cluster.vs["score"]))[i])
                                ]["original_id"]
                                if next_node not in self.center_nodes:
                                    self.center_nodes.append(next_node)
                            except IndexError:
                                pass
                        i += 1

                    if len(self.center_nodes) >= initial_partition_size:
                        self.center_nodes = self.center_nodes[:initial_partition_size]
            else:
                _graph: ig.Graph = self.graph.copy()
                match _partition_method:
                    case "random":
                        global_init_rng: np.random.Generator = np.random.default_rng(
                            seed=1
                        )
                        _graph.vs["score"] = global_init_rng.choice(
                            len(_graph.vs), len(_graph.vs), replace=False
                        ).tolist()
                    case "betweenness":
                        _graph.vs["score"] = _graph.betweenness()
                    case "closeness":
                        _graph.vs["score"] = _graph.closeness()

                global_init_i: int = 0

                while (
                    len(self.center_nodes) < initial_partition_size
                    or global_init_i < initial_partition_size
                ):
                    try:
                        next_node = _graph.vs[
                            int(np.flip(np.argsort(_graph.vs["score"]))[global_init_i])
                        ]["original_id"]
                        if next_node not in self.center_nodes:
                            self.center_nodes.append(next_node)
                    except IndexError:
                        pass
                    global_init_i += 1

                if len(self.center_nodes) >= initial_partition_size:
                    self.center_nodes = self.center_nodes[:initial_partition_size]

        self.initial_center_nodes: list[int] = self.center_nodes.copy()
        self.next_nodes: list[int] = []
        self.initial_partition_size: int = len(self.center_nodes)
        self.partition_size: int = len(self.center_nodes)
        self.max_partition_size: int = max_partition_size
        self.partition: ig.VertexClustering = ig.VertexClustering(
            self.graph,
            np.argmin(
                np.array(self.graph.distances(target=self.center_nodes)), axis=1
            ).tolist(),
        )

        self.mean_signal: npt.NDArray[np.floating] = np.zeros(
            (self.max_partition_size,)
            if signal.ndim == 1
            else (self.max_partition_size, signal.shape[1])
        )

        for i in range(self.initial_partition_size):
            if signal.ndim == 1:
                self.mean_signal[i] = signal[
                    np.flatnonzero(np.asarray(self.partition.membership) == i)
                ].mean()
            else:
                self.mean_signal[i] = signal[
                    np.flatnonzero(np.asarray(self.partition.membership) == i)
                ].mean(axis=0)

        self.wavelet_coefficients: npt.NDArray[np.floating] = np.zeros(
            (self.max_partition_size, 1, 2)
            if signal.ndim == 1
            else (self.max_partition_size, signal.shape[1], 2)
        )
        self.wavelet_coefficients[: self.initial_partition_size, :, 0] = (
            self.mean_signal[: self.initial_partition_size]
        )

        self.error: npt.NDArray[np.floating] = np.zeros(self.max_partition_size)
        coeff: float = 1 / len(self.graph.vs)
        if signal.ndim == 2:
            coeff /= signal.shape[1]

        for i in range(self.initial_partition_size):
            block: npt.NDArray[np.integer] = np.flatnonzero(
                np.asarray(self.partition.membership) == i
            )
            self.error[i] = (
                np.linalg.norm(
                    signal[block] - (self.mean_signal[i])
                    if signal.ndim == 1
                    else np.tile(self.mean_signal[i], (len(block), 1))
                )
                ** 2
                / coeff
            )

    def wedgelet_encode(
        self,
        signal: npt.NDArray[np.floating],
        method: (
            Literal[
                "fully_adaptive",
                "greedy",
                "j_centers",
                "closeness",
                "betweenness",
                "pagerank",
                "random",
            ]
            | None
        ) = None,
        method_parameter: int | None = None,
        tolerance: float = 1e-3,
        max_partition_size: int | None = None,
    ) -> None:
        if method not in [
            "fully_adaptive",
            "greedy",
            "j_centers",
            "closeness",
            "betweenness",
            "pagerank",
            "random",
            None,
        ]:
            raise ValueError(
                "Unsupported method. Supported methods: 'fully_adaptive', 'greedy', 'j_centers', 'closeness', 'betweenness', 'pagerank', 'random' (default)."
            )
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
            current_mean_signal: npt.NDArray[np.floating] = self.mean_signal[
                max_error_index
            ]
            current_partition: npt.NDArray[np.integer] = np.asarray(
                self.partition.membership
            )
            current_subgraph: ig.Graph = self.partition.subgraph(max_error_index)
            current_subgraph_original: npt.NDArray[np.integer] = np.array(
                current_subgraph.vs["original_id"]
            )

            center_index: int = int(
                np.flatnonzero(
                    np.asarray(current_subgraph.vs["original_id"]) == center_node
                )[0]
            )

            if method is None:
                _method = "random"
            else:
                _method = method

            _parameter: int = method_parameter if method_parameter is not None else 2
            sanitized_parameter: int = (
                _parameter
                if _parameter < len(current_subgraph.vs)
                else len(current_subgraph.vs) - 1
            )
            new_node_range: npt.NDArray[np.integer]

            match _method:
                case "random":
                    graph_encode_rng: np.random.Generator = np.random.default_rng(
                        seed=1
                    )
                    new_node_range = graph_encode_rng.choice(
                        np.arange(len(current_subgraph.vs))[
                            np.arange(len(current_subgraph.vs)) != center_index
                        ],
                        sanitized_parameter,
                        replace=False,
                    )
                case "fully_adaptive":
                    new_node_range = np.arange(len(current_subgraph.vs))[
                        np.arange(len(current_subgraph.vs)) != center_index
                    ]
                case "greedy":
                    new_node_range = np.argmax(
                        center_distance(
                            current_subgraph,
                            np.arange(len(current_subgraph.vs)).tolist(),
                            center_index,
                        ),
                        keepdims=True,
                    )
                case "j_centers":
                    new_node_range = j_centers(
                        current_subgraph,
                        sanitized_parameter,
                        input_centers=center_index,
                    )[0][1:]
                case "closeness":
                    current_closeness: npt.NDArray[np.floating] = np.asarray(
                        current_subgraph.closeness()
                    )
                    closeness_order: npt.NDArray[np.integer] = np.flip(
                        np.argsort(current_closeness)
                    )[: sanitized_parameter + 1]
                    new_node_range = closeness_order[closeness_order != center_index][
                        :sanitized_parameter
                    ]
                case "betweenness":
                    current_betweenness: npt.NDArray[np.floating] = np.asarray(
                        current_subgraph.betweenness()
                    )
                    betweenness_order: npt.NDArray[np.integer] = np.flip(
                        np.argsort(current_betweenness)
                    )[: sanitized_parameter + 1]
                    new_node_range = betweenness_order[
                        betweenness_order != center_index
                    ][:sanitized_parameter]
                case "pagerank":
                    current_pagerank: npt.NDArray[np.floating] = np.asarray(
                        current_subgraph.pagerank()
                    )
                    pagerank_order: npt.NDArray[np.integer] = np.flip(
                        np.argsort(current_pagerank)
                    )[: sanitized_parameter + 1]
                    new_node_range = pagerank_order[pagerank_order != center_index][
                        :sanitized_parameter
                    ]

            for possible_node in new_node_range:
                new_split: npt.NDArray[np.integer] = wedge_split(
                    current_subgraph, center_index, possible_node
                )

                cluster_1: npt.NDArray[np.integer] = current_subgraph_original[
                    np.flatnonzero(1 - new_split)
                ]
                cluster_2: npt.NDArray[np.integer] = current_subgraph_original[
                    np.flatnonzero(new_split)
                ]

                error_1: np.floating = (
                    (
                        np.linalg.norm(
                            signal[cluster_1]
                            - np.tile(
                                np.mean(signal[cluster_1], axis=0),
                                (
                                    (cluster_1.shape[0],)
                                    if signal.ndim == 1
                                    else (cluster_1.shape[0], 1)
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
                                    (cluster_2.shape[0],)
                                    if signal.ndim == 1
                                    else (cluster_2.shape[0], 1)
                                ),
                            )
                        )
                        ** 2
                    )
                    / signal.shape[0]
                    / (signal.shape[1] if signal.ndim > 1 else 1)
                )
                error_new: np.floating = error_1 + error_2

                if error_new <= max_error:
                    if len(self.center_nodes) == self.partition_size + 1:
                        self.center_nodes[-1] = current_subgraph_original[
                            int(possible_node)
                        ]
                    else:
                        self.center_nodes.append(
                            current_subgraph_original[int(possible_node)]
                        )

                    current_partition[cluster_1] = max_error_index
                    current_partition[cluster_2] = self.partition_size

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

                    self.error[max_error_index] = error_1
                    self.error[self.partition_size] = error_2
                    max_error = error_new

            max_error_index = int(np.argmax(self.error))
            max_error = self.error[max_error_index]

            self.partition_size += 1
            self.partition = ig.VertexClustering(self.graph, current_partition.tolist())

        if max_partition_size is None and self.partition_size < self.max_partition_size:
            self.mean_signal = self.mean_signal[: self.partition_size]
            self.wavelet_coefficients = self.wavelet_coefficients[: self.partition_size]


def center_distance(
    graph: ig.Graph, indices: list[int], center: int
) -> npt.NDArray[np.floating]:
    return np.array(graph.distances(source=indices, target=[center])).flatten()


def wedge_split(
    graph: ig.Graph, center_1: int, center_2: int
) -> npt.NDArray[np.integer]:
    return np.argmin(np.array(graph.distances(target=[center_1, center_2])), axis=1)


def j_centers(
    g: ig.Graph,
    n_clusters: int,
    greedy_search_subset: npt.ArrayLike | None = None,
    input_centers: npt.ArrayLike | np.integer | int | None = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    N: int = len(g.vs)

    input_center: npt.NDArray[np.integer]
    if input_centers is None:
        rng = np.random.default_rng(seed=1)
        input_center = rng.integers(N, size=(1,))
    elif isinstance(input_centers, int) or isinstance(input_centers, np.integer):
        input_center = np.array([input_centers])
    else:
        input_center = np.array(input_centers)

    greedy_subset: npt.NDArray[np.integer]
    if greedy_search_subset is None:
        greedy_subset = np.arange(N)
    else:
        greedy_subset = np.array(greedy_search_subset)

    distances: npt.NDArray[np.floating] = np.zeros((n_clusters, N))
    centers: npt.NDArray[np.integer] = np.repeat(-1, n_clusters)
    centers[: input_center.shape[0]] = input_center

    for j in range(n_clusters - 1):
        distances[j] = center_distance(g, g.vs, centers[j])
        if j >= input_center.shape[0] - 1:
            centers[j + 1] = greedy_subset[
                np.argmax(np.min(distances[: j + 1, greedy_subset], axis=0))
            ]
    distances[-1] = center_distance(g, g.vs, centers[-1])

    return centers, distances.argmin(axis=0)
