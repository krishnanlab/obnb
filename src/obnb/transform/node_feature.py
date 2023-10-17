"""Node feature transformation module."""
import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

import obnb
from obnb.feature import FeatureVec
from obnb.registry import register_nodefeat
from obnb.transform.base import BaseDatasetTransform
from obnb.typing import Optional
from obnb.util.logger import display_pbar
from obnb.util.misc import get_num_workers


class BaseNodeFeatureTransform(BaseDatasetTransform, ABC):
    """Base node feature transformation abstract class.

    Specific implementation should overwrite the :meth:`_prepare_feat` method,
    which computes the node features given the constructed dataset object.

    """

    NAME_PREFIX = "nodefeat"

    def __init__(self, dim: int, as_feature: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.as_feature = as_feature

    def __call__(self, dataset):
        self.logger.info(f"Computing {self.name} features")
        feat = self._prepare_feat(dataset)
        dataset.update_extras(self.fullname, feat)

        if self.as_feature:
            if dataset.feature is not None:
                warnings.warn(
                    "Node features already exist in dataset, overwritting "
                    f"with {self.name}. Please make sure node features is "
                    "empty to suppress this message.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            node_ids = list(dataset.graph.node_ids)
            dataset.feature = FeatureVec.from_mat(feat, node_ids)

    @abstractmethod
    def _prepare_feat(self, dataset) -> np.ndarray:
        ...


@register_nodefeat
class OneHotLogDeg(BaseNodeFeatureTransform):
    """One-hot log degree feature."""

    def _prepare_feat(self, dataset):
        log_deg = np.log(dataset.get_adj().sum(axis=1, keepdims=True))
        feat = KBinsDiscretizer(
            n_bins=self.dim,
            encode="onehot-dense",
            strategy="uniform",
        ).fit_transform(log_deg)
        self.logger.info(f"Bins stats:\n{feat.sum(0)}")
        return feat


@register_nodefeat
class Constant(BaseNodeFeatureTransform):
    """Constant feature."""

    def _prepare_feat(self, dataset):
        if self.dim != 1:
            warnings.warn(
                f"Constant feature only allows dim of 1, got {self.dim!r}. "
                "Implicitly setting dim to 1. Please update dim setting to "
                "resolve this message.",
                UserWarning,
                stacklevel=2,
            )
        return np.ones((dataset.get_adj().shape[0], 1))


@register_nodefeat
class RandomNormal(BaseNodeFeatureTransform):
    """Random features drawn from standard normal distribution."""

    def _prepare_feat(self, dataset):
        rng = np.random.default_rng(self.random_state)
        return rng.random((dataset.size, self.dim))


@register_nodefeat
class Orbital(BaseNodeFeatureTransform):
    """Graphlet feature."""

    def __init__(
        self,
        *args,
        graphlet_size: int = 3,
        num_workers: int = -1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.graphlet_size = graphlet_size
        self.num_workers = get_num_workers(num_workers)

    def _prepare_feat(self, dataset):
        from obnb.ext.orbital_features import orbital_feat_extract

        return orbital_feat_extract(
            dataset.get_sparse_graph(),
            n_jobs=self.num_workers,
            as_array=True,
            verbose=display_pbar(self.log_level),
        )


@register_nodefeat
class SVD(BaseNodeFeatureTransform):
    """Adjacency matrix SVD feature."""

    def _prepare_feat(self, dataset):
        sparse_adj = sp.csr_matrix(dataset.get_adj())
        feat, _, _ = sp.linalg.svds(sparse_adj, k=self.dim, which="LM")
        # Work around for potential negative strides by making a copy of the array
        # https://discuss.pytorch.org/t/negative-strides-in-tensor-error/134287/2
        return feat.copy()


@register_nodefeat
class LapEigMap(BaseNodeFeatureTransform):
    """Graph Laplacian eigenmap feature."""

    def _prepare_feat(self, dataset):
        adj = dataset.get_adj()

        sparse_lap = sp.csr_matrix(np.diag(adj.sum(1)) - adj)
        if (sparse_lap != sparse_lap.T).sum() != 0:
            raise ValueError("The input network must be undirected.")

        # Symmetric normalized graph Laplacian
        deg_inv_sqrt = sp.diags(1 / np.sqrt(adj.sum(1)))
        sparse_lap = deg_inv_sqrt @ sparse_lap @ deg_inv_sqrt

        eigvals, eigvecs = sp.linalg.eigsh(sparse_lap, which="SM", k=self.dim + 1)
        sorted_idx = eigvals.argsort()
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        assert (
            eigvals[1:] > 1e-8
        ).all(), f"Network appears to be disconnected.\n{eigvals=}"
        feat = eigvecs[:, 1:] / np.sqrt((eigvecs[:, 1:] ** 2).sum(0))  # l2 normalize

        return feat


@register_nodefeat
class RandomWalkDiag(BaseNodeFeatureTransform):
    """Random walk diagonals feature."""

    def _prepare_feat(self, dataset):
        adj = dataset.get_adj()

        p_mat = adj / adj.sum(0)
        feat = np.zeros((adj.shape[0], self.dim))
        vec = np.ones(adj.shape[0])
        for i in range(self.dim):
            vec = p_mat @ vec
            feat[:, i] = vec

        return feat


@register_nodefeat
class RandProjGaussian(BaseNodeFeatureTransform):
    """Adjacency matrix gaussian random projection feature."""

    def _prepare_feat(self, dataset):
        grp = GaussianRandomProjection(
            n_components=self.dim,
            random_state=self.random_state,
        )
        return grp.fit_transform(dataset.get_adj())


@register_nodefeat
class RandProjSparse(BaseNodeFeatureTransform):
    """Adjacency matrix sparse random projection feature."""

    def _prepare_feat(self, dataset):
        srp = SparseRandomProjection(
            n_components=self.dim,
            dense_output=True,
            random_state=self.random_state,
        )
        return srp.fit_transform(dataset.get_adj())


@register_nodefeat
class LINE1(BaseNodeFeatureTransform):
    """First order LINE embedding feature."""

    def _prepare_feat(self, dataset):
        from obnb.ext.grape import grape_embed

        feat = grape_embed(
            dataset.get_sparse_graph(),
            "FirstOrderLINEEnsmallen",
            dim=self.dim,
            as_array=True,
            random_state=self.random_state,
            verbose=display_pbar(self.log_level),
        )
        return feat


@register_nodefeat
class LINE2(BaseNodeFeatureTransform):
    """Second order LINE embedding feature."""

    def _prepare_feat(self, dataset):
        from obnb.ext.grape import grape_embed

        feat = grape_embed(
            dataset.get_sparse_graph(),
            "SecondOrderLINEEnsmallen",
            dim=self.dim,
            as_array=True,
            random_state=self.random_state,
            verbose=display_pbar(self.log_level),
        )
        return feat


@register_nodefeat
class Node2vec(BaseNodeFeatureTransform):
    """Node2vec(+) embedding feature."""

    def __init__(
        self,
        *args,
        p: float = 1,
        q: float = 1,
        extend: bool = False,
        gamma: float = 0,
        num_walks: int = 10,
        walk_length: int = 80,
        window_size: int = 10,
        epochs: int = 1,
        num_workers: int = -1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p = p
        self.q = q
        self.extend = extend
        self.gamma = gamma
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.epochs = epochs
        self.num_workers = get_num_workers(num_workers)

    def _prepare_feat(self, dataset):
        from obnb.ext.pecanpy import pecanpy_embed

        feat = pecanpy_embed(
            dataset.get_sparse_graph(),
            dim=self.dim,
            p=self.q,
            q=self.q,
            extend=self.extend,
            gamma=self.gamma,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            window_size=self.window_size,
            epochs=self.epochs,
            as_array=True,
            workers=self.num_workers,
            verbose=display_pbar(self.log_level),
            random_state=self.random_state,
        )
        return feat


@register_nodefeat
class Walklets(BaseNodeFeatureTransform):
    """Walklets embedding feature."""

    def __init__(
        self,
        *args,
        epochs: int = 1,
        window_size: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self.window_size = window_size

    def _prepare_feat(self, dataset):
        # NOTE: The resulding feat is a concatenation of (window_size x 2) number
        # of embeddings, each has the dimension of dim.
        from obnb.ext.grape import grape_embed

        feat_raw = grape_embed(
            dataset.get_sparse_graph(),
            "WalkletsSkipGramEnsmallen",
            dim=self.dim * self.window_size,  # one emb per-window (both-sides)
            as_array=True,
            grape_enable=True,
            random_state=self.random_state,
            epochs=self.epochs,
        )

        # Reduce multscale embedding to dim via PCA following arxiv:1605.02115
        if feat_raw.shape[1] > self.dim:
            pca = PCA(n_components=self.dim, random_state=self.random_state)
            feat = pca.fit_transform(feat_raw)
            evr = pca.explained_variance_ratio_.sum()
            obnb.logger.info(
                "Reduced concatenated walklets embedding dimensions from "
                f"{feat_raw.shape[1]} to {feat.shape[1]} (EVR={evr:.2%}).",
            )
        else:
            feat = feat_raw

        return feat


@register_nodefeat
class AttnWalk(BaseNodeFeatureTransform):
    """Attention walk embedding feature."""

    def __init__(
        self,
        *args,
        walk_length: int = 80,
        window_size: int = 5,
        beta: float = 0.5,
        gamma: float = 0.0,
        epochs: int = 200,
        lr: float = 0.01,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.walk_length = walk_length
        self.window_size = window_size
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def _prepare_feat(self, dataset):
        from obnb.ext.attnwalk import attnwalk_embed

        feat, attn = attnwalk_embed(
            dataset.get_sparse_graph(),
            dim=self.dim,
            walk_length=self.walk_length,
            window_size=self.window_size,
            beta=self.beta,
            gamma=self.gamma,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            verbose=display_pbar(self.log_level),
            as_array=True,
            return_attn=True,
        )
        attn_str = ", ".join(f"{i:.4f}" for i in attn)
        obnb.logger.info(f"AttnWalk attentions: [{attn_str}]")
        return feat


@register_nodefeat
class Adj(BaseNodeFeatureTransform):
    """Adjacency matrix feature."""

    def __init__(self, *args, dim: Optional[int] = None, **kwargs):
        if dim is not None:
            warnings.warn(
                "Adj node features do not use the dim argument. "
                "Please remove to suppress this message.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(
            *args,
            dim=0,
            **kwargs,
        )  # type: ignore  # incorrect diagnostic of multiple dim input

    def _prepare_feat(self, dataset):
        return dataset.get_adj().copy()
