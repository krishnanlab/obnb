"""Grpahlet orbital feature extraction."""
import itertools
import multiprocessing
from functools import partial

import networkx as nx
import pandas as pd
from networkx.generators.atlas import graph_atlas_g
from tqdm import tqdm

from nleval import logger


class OrbitCountingMachine:
    """Connected motif orbital role counter.

    Adpated from: https://github.com/benedekrozemberczki/OrbitalFeatures

    This modified implementation added multiprocessing parallelization to the
    feature extraction process.

    Reference
    ---------
    Pržulj, Nataša. "Biological network comparison using graphlet degree
    distribution." Bioinformatics (2007)

    """

    def __init__(self, graph, graphlet_size: int = 4, n_jobs: int = 1):
        """Creating an orbital role counter machine.

        Args:
            graph: A NetowrkX graph object.
            graphlet_size: The size of the graphlets to extract.
            n_jobs: Number of parallel jobs to run. If -1, use all available
                CPU counts.

        """
        self.graph = graph
        self.graphlet_size = graphlet_size
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

    def create_edge_subsets(self):
        """Enumerating connected subgraphs with size 2 up to the graphlet size."""
        logger.info("    Enumerating subgraphs.")
        self.edge_subsets = dict()
        subsets = [[edge[0], edge[1]] for edge in self.graph.edges()]
        self.edge_subsets[2] = subsets
        unique_subsets = dict()
        for i in range(3, self.graphlet_size + 1):
            logger.info("    Enumerating graphlets with size: {i}")
            for subset in tqdm(subsets):
                for node in subset:
                    for neb in self.graph.neighbors(node):
                        new_subset = subset + [neb]
                        if len(set(new_subset)) == i:
                            new_subset.sort()
                            unique_subsets[tuple(new_subset)] = 1
            subsets = [list(k) for k, v in unique_subsets.items()]
            self.edge_subsets[i] = subsets
            unique_subsets = dict()

    def enumerate_graphs(self):
        """Creating a hash table of the benchmark motifs."""
        graphs = graph_atlas_g()
        self.interesting_graphs = {i: [] for i in range(2, self.graphlet_size + 1)}
        for graph in graphs:
            if (
                (graph.number_of_nodes() > 1)
                and (graph.number_of_nodes() <= self.graphlet_size)
                and nx.is_connected(graph)
            ):
                self.interesting_graphs[graph.number_of_nodes()].append(graph)

    def enumerate_categories(self):
        """Creating a hash table of benchmark orbital roles."""
        main_index = 0
        self.categories = dict()
        for size, graphs in self.interesting_graphs.items():
            self.categories[size] = dict()
            for index, graph in enumerate(graphs):
                self.categories[size][index] = dict()
                degrees = list({graph.degree(node) for node in graph.nodes()})
                for degree in degrees:
                    self.categories[size][index][degree] = main_index
                    main_index = main_index + 1
        self.unique_motif_count = main_index + 1

    def setup_features(self):
        """Counting all the orbital roles."""
        logger.info("    Counting orbital roles.")
        self.features = {
            node: {i: 0 for i in range(self.unique_motif_count)}
            for node in self.graph.nodes()
        }
        for size, node_lists in self.edge_subsets.items():
            graphs = self.interesting_graphs[size]
            _wrapped_setup_feature = partial(
                self._setup_feature,
                graph=self.graph,
                size=size,
                graphs=graphs,
            )
            with multiprocessing.Pool(self.n_jobs) as p:
                feat_lists = p.map(_wrapped_setup_feature, tqdm(node_lists))
            self._feat_lists_to_features(feat_lists)

    @staticmethod
    def _setup_feature(nodes, graph, size, graphs):
        # sub_gr = self.graph.subgraph(nodes)
        sub_gr = graph.subgraph(nodes)
        out_list = []
        for index, graph in enumerate(graphs):
            if nx.is_isomorphic(sub_gr, graph):
                for node in sub_gr.nodes():
                    out_list.append((node, size, index, sub_gr.degree(node)))
                break
        return out_list

    def _feat_lists_to_features(self, feat_lists):
        for node, size, index, degree in itertools.chain(*feat_lists):
            self.features[node][self.categories[size][index][degree]] += 1

    def create_tabular_motifs(self):
        """Creating a table with the orbital role features."""
        motifs_counts_lists = [
            [self.features[n][i] for i in range(self.unique_motif_count)]
            for n in self.graph.nodes()
        ]
        self.motifs = pd.DataFrame(
            motifs_counts_lists,
            index=self.graph.nodes(),
            columns=[f"role_{i}" for i in range(self.unique_motif_count)],
        )

    def extract_features(self) -> pd.DataFrame:
        """Executing steps for feature extraction."""
        logger.info("[*] Begin extracting orbital features.")
        self.create_edge_subsets()
        self.enumerate_graphs()
        self.enumerate_categories()
        self.setup_features()
        self.create_tabular_motifs()
        return self.motifs
