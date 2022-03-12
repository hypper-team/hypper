import pandas as pd
import numpy as np
import sys

from bidict import bidict
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import diags, csr_matrix, csc_matrix

# TODO
#  * caching mechanism for hypergraph weighting
#  * include weights in semi-random walk path selection

class HyperGraph:
    """Hypergraph representation of an input dataset.

    HyperGraph stores representation in sparse incidence matrix based on `pandas.DataFrame` input data structure. HyperGraph class implements also class-dependent weighting method.
    """
    def __init__(self, input_data: pd.DataFrame, label: str, random_seed=42, verbosity=None):
        """
        Args:
            input_data (pandas.DataFrame): Input data.
            label (str, optional): Label column name.
            random_seed (int, optional): Random seed.
            verbosity (int, optional): Value greater than 0 displays info about process progress. Defaults to None.

        Raises:
            ValueError: Not recognized format of an input data.
        """

        if label == 'class': raise ValueError("Label shouldn't be named `class` due to Pandas internal error")

        # input_data = input_data.astype(str)
        self.X, self.y = input_data.drop(label, axis=1), input_data[label]
        self.random_seed = random_seed
        self.label = label
        self.verbosity = verbosity

        if self.verbosity:
            print(f"Input file shape: {input_data.shape} ...")
            print(f"Input file in memory size: {sys.getsizeof(self.input_data)*0.000001} MB ...")

        if isinstance(input_data, pd.DataFrame):
            self._create_hypergraph_representation() # Generates: self.incidence_matrix, self.edges, self.vertices
            if self.verbosity: print("Created hypergraph-based data representation ...")
        else:
            raise ValueError("Not recognized input format.")

    def _create_hypergraph_representation(self):
        """Function generater hypergraph representation for the input data. This representation takes form of a incidence matrix and indices to vertices/hyperedges names mapping.
        """
        ohe = OneHotEncoder(drop = None, sparse=True, handle_unknown='error')
        self.incidence_matrix = ohe.fit_transform(self.X.values)
        self.incidence_matrix.multiply(csr_matrix(1/np.sqrt(np.sum(self.incidence_matrix, axis=0))))

        edges_names = ohe.get_feature_names_out(self.X.columns)
        self.edges = bidict(zip(edges_names, list(range(0, len(edges_names)))))

        vertices_names = self.X.index.values.tolist()
        self.vertices = bidict(zip(vertices_names, list(range(0, len(vertices_names)))))

        ohe = OneHotEncoder(drop = None, sparse=True, handle_unknown='error')
        self.vertices_weights = ohe.fit_transform(self.y.values.reshape(-1, 1))
        edges_labels_names = ohe.get_feature_names_out([self.label])
        self.edges_labels = bidict(zip(edges_labels_names, list(range(0, len(edges_labels_names)))))

    def calculate_weights(self, iterations, normalization_strategy='max', iter_history=False):
        """Calculates matrices with columns corresponding to classes, and rows corresponding to hyperedges/vertices. The results are class-dependent hypergraph weights for hyperedges and vertices.

        Args:
            iterations (int): Number of hypergraph weighting iterations.
            normalization_strategy (str, optional): Normalization strategy. Defaults to 'max'. Options: 'max', 'l1', 'l2'.
            iter_history (boolean, optional): If True rememebers hyperedges and vertices weights from every iteration.

        Returns:
            list: If `iter_history` is set to `True` returns list of tuples, where evry tuple consists of hyperedges and vertices weight matrices for subsequent iterations.
        """
        # Weights initialization basend on class distribution
        self.vertices_weights = self.vertices_weights.multiply(csr_matrix(np.sum(self.vertices_weights, axis=0)/self.vertices_weights.shape[0]))

        if iter_history: iteration_history = []
        # Iterations for weight calculations
        for _ in range(0, iterations):
            # Calculating vertex dependent hyperedges weights for every class
            self.edges_weights = self.vertices_weights.T.dot(self.incidence_matrix)
            # Hyperedges weights normalization
            self.edges_weights = self.normalize(self.edges_weights, normalization_strategy, axis=1)

            self.vertices_weights = self.incidence_matrix.dot(self.edges_weights.T)
            self.vertices_weights = self.normalize(self.vertices_weights.T, normalization_strategy, axis=1).T

            if iter_history: iteration_history.append((self.edges_weights, self.vertices_weights))
        if iter_history: 
            self.weighting_iteration_history = iteration_history
        if self.verbosity: print("Class-dependent weight calculated ...")
        
    def normalize(self, matrix: csc_matrix, normalization_strategy: str, axis: int):
        """Method implements three normalization strategies: `Max`, `L1`, and `L2`.

        Args:
            matrix (csc_matrix): Input sparse matrix.
            normalization_strategy (str): Type of the normalization strategy.
            axis (int): Axis.

        Raises:
            SyntaxError: Not recognized normalization strategy.

        Returns:
            scipy.sparse.csc_matrix: Normalized sparse matrix.
        """
        if normalization_strategy == 'max': return diags(1/matrix.max(axis=axis).A.ravel()).dot(matrix)
        elif normalization_strategy =='l1': return diags(1/matrix.sum(axis=axis).A.ravel()).dot(matrix)
        elif normalization_strategy == 'l2': return diags(1/np.sqrt(matrix.power(2).sum(axis=axis)).A.ravel()).dot(matrix)
        else: raise SyntaxError(f'Unknown normalization strategy: {normalization_strategy}')
