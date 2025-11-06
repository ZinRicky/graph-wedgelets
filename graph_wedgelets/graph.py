# Necessary libraries
import numpy as np
import igraph as ig
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

        self.graph["original_id"] = [x.index for x in self.graph.vs]

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

            _leiden: bool = use_leiden or initial_partition_size is None

            if _leiden:
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
                        next_node = cluster.vs[
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
        self.initial_partition_size: int = len(self.center_nodes)
        self.max_partition_size: int = max_partition_size


def center_distance(
    graph: ig.Graph, indices: list[int], center: int
) -> npt.NDArray[np.floating]:
    return np.array(graph.distances(source=indices, target=[center])).flatten()


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
