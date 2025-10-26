import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import random


def easing(x: float) -> float:
    return 1 - (1 - x) ** 2


def center_distance(graph, indices, center):
    return np.array(graph.distances(source=indices, target=[center])).flatten()


def _main(x: int) -> None:
    match x:
        case 1:
            # Basic graph visualisation
            G = ig.Graph.Erdos_Renyi(n=2000, p=0.01)
            G_1 = G.connected_components().giant()
            G_1.vs["harmonic_centrality"] = G_1.harmonic_centrality()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect(1, anchor="C")
            ig.plot(
                G_1,
                layout="kk",
                target=ax,
                vertex_size=[30 * easing(x) for x in G_1.vs["harmonic_centrality"]],
            )
            plt.show()
        case 2:
            # Preliminary study of igraph
            N = 50
            G = ig.Graph.Erdos_Renyi(n=N, p=0.01)
            G_1 = G.connected_components().giant()
            print(f"Vertices: {len(G_1.vs)}\t({len(G_1.vs)/N:.2%})")
            # print(
            #     G_1.community_leiden(
            #         objective_function="modularity", n_iterations=-1
            #     ).giant()
            # )
            print(G_1)
            A = sps.csr_array(G_1.get_adjacency_sparse())
            q = np.zeros(A.shape[0], dtype=np.int64)
            q[0] = 1
            print(A.toarray())
            print((A @ q) / np.linalg.norm(A @ q, ord=1))
        case 3:
            # Preliminary study of J-centers
            N_vertices = 1000
            probability = 0.01
            random.seed(1)
            G = ig.Graph.Erdos_Renyi(n=N_vertices, p=probability)
            G_1 = G.connected_components().giant()
            G_1.vs["original_id"] = [v.index for v in G_1.vs]
            print(f"Vertices: {len(G_1.vs)}\t({len(G_1.vs)/N_vertices:.2%})")
            leid = G_1.community_leiden(
                objective_function="modularity", n_iterations=-1
            )
            G_2 = leid.giant()
            # print(np.array([[v.index for v in G_2.vs], G_2.vs["original_id"]]))

            J = 2

            A = sps.csr_array(G_1.get_adjacency_sparse())
            N = A.shape[0]
            D = np.zeros((J, N))

            rng = np.random.default_rng(seed=1)

            default_centers: list[int] = rng.choice(G_2.vs["original_id"], 1).tolist()
            print(f"{default_centers=}")
            n_d_centers = len(default_centers)

            cluster_centers: list[int] = [-1 for _ in range(J)]
            cluster_centers[:n_d_centers] = default_centers

            for j in range(J - 1):
                D[j, G_2.vs["original_id"]] = center_distance(
                    G_1, G_2.vs["original_id"], cluster_centers[j]
                )
                # print(center_distance(G_1, G_2.vs["original_id"], cluster_centers[j]))
                # quit()
                if j >= len(default_centers) - 1:
                    cluster_centers[j + 1] = int(
                        G_2.vs["original_id"][
                            np.argmax(np.min(D[: j + 1, G_2.vs["original_id"]], axis=0))
                        ]
                    )
            D[-1, G_2.vs["original_id"]] = center_distance(
                G_1, G_2.vs["original_id"], cluster_centers[-1]
            )

            print(f"{cluster_centers=}")
            clusters = D.argmin(axis=0)
            print(f"{clusters=}")


if __name__ == "__main__":
    _main(3)
