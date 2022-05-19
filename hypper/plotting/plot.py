import numpy as np
import hypernetx

from collections import defaultdict
from hypper.hypergraph import HyperGraph


class HNX_Hypergraph(hypernetx.Hypergraph):
    """Hypper wrapper for HyperNetX hypergraph representation.

    Hypper hypergraph representation can be visualized thanks to HyperNetX library (**https://github.com/pnnl/HyperNetX**).
    """

    def __init__(self, hg: HyperGraph, include_label: bool = False):
        """Creates hypergraph representation with HyperNetX library.
        Args:
            hg (hypper.hypergraph.HyperGraph): Hypper HyperGraph object.
            include_label: `True` if label should be visible as one of the hyperedges. Defaults to `False`.
        """
        self.include_label = include_label
        super().__init__(self._prepare_structure(hg))

    def _prepare_structure(self, hg: HyperGraph) -> hypernetx.Hypergraph:
        new_representation = defaultdict(list)
        nonzero_idx = hg.incidence_matrix.nonzero()
        for vertex_idx, edge_idx in zip(nonzero_idx[0], nonzero_idx[1]):
            new_representation[hg.edges.inverse[edge_idx]].append(
                hg.vertices.inverse[vertex_idx]
            )
        if self.include_label:
            for row_idx, row in enumerate(hg.vertices_weights):
                new_representation[hg.edges_labels.inverse[np.argmax(row)]].append(
                    hg.vertices.inverse[row_idx]
                )
        return new_representation

    def draw(self, **kwargs) -> None:
        """Method uses hypernetx.draw to plot hypergraph."""
        hypernetx.draw(self, **kwargs)

    def draw_collapse_nodes(self, **kwargs) -> None:
        """Method uses hypernetx.draw with `collapsed_nodes` parameter to plot hypergraph."""
        hypernetx.draw(self.collapse_nodes(), **kwargs)
