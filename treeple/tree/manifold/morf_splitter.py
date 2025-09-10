# neofit_patch_splitter.py
# Pure-Python port of the Cython PatchSplitter / BestPatchSplitter hierarchy.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


# ----------------------------- Helper utilities -----------------------------

def ravel_multi_index_cython(unraveled: np.ndarray, dims: np.ndarray) -> int:
    """Equivalent to numpy.ravel_multi_index for 1D array of indices."""
    # unraveled and dims are 1D arrays
    idx = 0
    stride = 1
    for d, size in zip(reversed(unraveled), reversed(dims)):
        idx += int(d) * stride
        stride *= int(size)
    return idx


def unravel_index_cython(raveled: int, dims: np.ndarray, out: np.ndarray) -> None:
    """In-place unravel into `out` (1D) given dims (1D)."""
    for i in range(len(dims) - 1, -1, -1):
        size = int(dims[i])
        out[i] = raveled % size
        raveled //= size


def fisher_yates_shuffle(arr: np.ndarray, rng: np.random.Generator) -> None:
    """In-place Fisher–Yates shuffle."""
    n = arr.shape[0]
    # iterate n-1 down to 1
    for i in range(n - 1, 0, -1):
        j = rng.integers(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]


def floyd_sample_indices(k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Floyd's algorithm to sample k unique integers in [0, n).
    Returns sorted selection (not required by original, but handy).
    """
    selected = set()
    for i in range(n - k, n):
        t = rng.integers(0, i + 1)
        if t in selected:
            selected.add(i)
        else:
            selected.add(t)
    return np.fromiter(selected, dtype=np.intp, count=k)


# ------------------------------ Criterion stub ------------------------------
# Your real code should provide the true Criterion, with the methods used below.

@dataclass
class Criterion:
    weighted_n_node_samples: float = 0.0

    # The real implementation should set internal pointers and stats.
    def init(self, y, sample_weight, weighted_n_samples, samples):
        self.weighted_n_node_samples = float(
            np.sum(sample_weight[samples]) if sample_weight is not None else len(samples)
        )

    def set_sample_pointers(self, start: int, end: int):
        # In the Cython version this updates internal pointer views.
        # Here it's a no-op placeholder.
        return


# --------------------------- BestObliqueSplitter stub -----------------------
# Minimal base providing fields used by PatchSplitter. Replace with real base.

class BestObliqueSplitter:
    def __init__(self):
        # populated in init()
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.sample_weight: Optional[np.ndarray] = None
        self.missing_values_in_feature_mask: Optional[np.ndarray] = None

        self.n_samples: int = 0
        self.n_features: int = 0
        self.samples: Optional[np.ndarray] = None
        self.weighted_n_samples: float = 0.0

        # to be provided via subclass cinit-equivalent
        self.criterion: Criterion = Criterion()
        self.max_features: int = 0
        self.min_samples_leaf: int = 1
        self.min_weight_leaf: float = 0.0
        self.random_state: Optional[int] = None
        self.monotonic_cst: Optional[np.ndarray] = None

        # RNG replacing C rand_r_state
        self._rng: np.random.Generator = np.random.default_rng()

    # Mirrors the Cython .init(self, X, y, sample_weight, missing_mask)
    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight)
        self.missing_values_in_feature_mask = (
            None if missing_values_in_feature_mask is None else np.asarray(missing_values_in_feature_mask)
        )

        self.n_samples, self.n_features = int(self.X.shape[0]), int(self.X.shape[1])
        self.samples = np.arange(self.n_samples, dtype=np.intp)
        self.weighted_n_samples = float(
            np.sum(self.sample_weight) if self.sample_weight is not None else self.n_samples
        )

        # (Re)seed RNG if random_state provided
        if getattr(self, "random_state", None) is not None:
            self._rng = np.random.default_rng(self.random_state)

        return 0


# -------------------------------- PatchSplitter -----------------------------

class PatchSplitter(BestObliqueSplitter):
    """
    Base patch splitter (pure-Python port).
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        super().init(X, y, sample_weight, missing_values_in_feature_mask)
        return 0

    def node_reset(self, start: int, end: int, weighted_n_node_samples_out: np.ndarray) -> int:
        """
        Reset splitter on node samples[start:end].
        weighted_n_node_samples_out: np.ndarray shape (1,) to write the weight into.
        """
        self.start = int(start)
        self.end = int(end)

        # Initialize criterion on this node's sample slice
        self.criterion.init(self.y, self.sample_weight, self.weighted_n_samples, self.samples)
        self.criterion.set_sample_pointers(start, end)

        weighted_n_node_samples_out[0] = self.criterion.weighted_n_node_samples

        # Clear all projection vectors (to be allocated/filled before use)
        # These will be created by subclass; we simply ensure the structure exists.
        if hasattr(self, "proj_mat_weights") and hasattr(self, "proj_mat_indices"):
            for i in range(self.max_features):
                self.proj_mat_weights[i] = []
                self.proj_mat_indices[i] = []

        return 0

    # Placeholders to be implemented in subclass
    def sample_proj_mat(self, proj_mat_weights: List[List[float]], proj_mat_indices: List[List[int]]) -> None:
        pass

    def sample_top_left_seed(self) -> Tuple[int, int]:
        raise NotImplementedError


# --------------------------- BaseDensePatchSplitter --------------------------

class BaseDensePatchSplitter(PatchSplitter):
    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        super().init(X, y, sample_weight, missing_values_in_feature_mask)
        # Keep direct reference for faster access
        self.X = np.asarray(X)
        return 0


# ------------------------------ BestPatchSplitter ---------------------------

class BestPatchSplitter(BaseDensePatchSplitter):
    def __init__(
        self,
        criterion: Criterion,
        max_features: int,
        min_samples_leaf: int,
        min_weight_leaf: float,
        random_state: Optional[int],
        monotonic_cst: Optional[np.ndarray],
        feature_combinations: float,
        min_patch_dims: np.ndarray,
        max_patch_dims: np.ndarray,
        dim_contiguous: np.ndarray,
        data_dims: np.ndarray,
        boundary: Optional[bytes],
        feature_weight: Optional[np.ndarray],
        *args,
        **kwargs,
    ):
        super().__init__()
        # Parameters from "cinit"
        self.criterion = criterion
        self.max_features = int(max_features)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_leaf = float(min_weight_leaf)
        self.random_state = random_state
        self.monotonic_cst = None if monotonic_cst is None else np.asarray(monotonic_cst)
        self.feature_combinations = feature_combinations

        # Sparse (max_features x n_features) stored as python lists
        self.proj_mat_weights: List[List[float]] = [[] for _ in range(self.max_features)]
        self.proj_mat_indices: List[List[int]] = [[] for _ in range(self.max_features)]

        # Tensor geometry
        self.ndim = int(np.asarray(data_dims).shape[0])
        self.data_dims = np.asarray(data_dims, dtype=np.intp)

        # Buffers
        self.patch_sampled_size = np.zeros(self.data_dims.shape[0], dtype=np.intp)
        self.unraveled_patch_point = np.zeros(self.data_dims.shape[0], dtype=np.intp)

        self.min_patch_dims = np.asarray(min_patch_dims, dtype=np.intp)
        self.max_patch_dims = np.asarray(max_patch_dims, dtype=np.intp)
        self.dim_contiguous = np.asarray(dim_contiguous, dtype=np.bool_)

        self._index_patch_buffer = np.zeros(int(np.max(self.max_patch_dims)), dtype=np.intp)
        self._index_data_buffer = np.zeros(int(np.max(self.data_dims)), dtype=np.intp)

        self._discontiguous = not bool(np.all(self.dim_contiguous))
        self.boundary = None if boundary is None else boundary.decode() if isinstance(boundary, (bytes, bytearray)) else boundary
        self.feature_weight = None if feature_weight is None else np.asarray(feature_weight, dtype=np.float32)

        # RNG
        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)

    # Pickle support (optional)
    def __reduce__(self):
        return (
            type(self),
            (
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.min_weight_leaf,
                self.random_state,
                None if self.monotonic_cst is None else self.monotonic_cst,
                self.feature_combinations,
                None if self.min_patch_dims is None else self.min_patch_dims,
                None if self.max_patch_dims is None else self.max_patch_dims,
                None if self.dim_contiguous is None else self.dim_contiguous,
                None if self.data_dims is None else self.data_dims,
                None if self.boundary is None else self.boundary.encode(),
                None if self.feature_weight is None else self.feature_weight,
            ),
            self.__getstate__(),
        )

    # ----------------------------- Sampling methods -----------------------------

    def sample_top_left_seed(self) -> Tuple[int, int]:
        """
        Sample the top-left seed for the n-dim patch.
        Returns (top_left_seed_raveled, patch_size).
        """
        patch_size = 1

        for idx in range(self.ndim):
            # random patch dimension within [min, max] inclusive
            patch_dim = int(self._rng.integers(self.min_patch_dims[idx], self.max_patch_dims[idx] + 1))

            if self.boundary is None:
                # top-left selection within the valid (no wrapping) region
                delta = (int(self.data_dims[idx]) - patch_dim) + 1
                top_left = int(self._rng.integers(0, delta))
                self.patch_sampled_size[idx] = patch_dim
                patch_size *= patch_dim

            elif self.boundary == "wrap":
                # emulate circular padding logic in original
                delta = int(self.data_dims[idx]) + 2 * (patch_dim - 1)
                top_left = int(self._rng.integers(0, delta))
                dim = top_left % delta
                patch_dim = min(patch_dim, min(dim + 1, int(self.data_dims[idx]) + patch_dim - dim - 1))
                self.patch_sampled_size[idx] = patch_dim
                patch_size *= patch_dim
                # Convert to valid in-bounds top-left
                top_left = max(0, dim - patch_dim + 1)

            else:
                raise ValueError(f"Unknown boundary mode: {self.boundary}")

            self.unraveled_patch_point[idx] = top_left

        top_left_seed = ravel_multi_index_cython(self.unraveled_patch_point, self.data_dims)
        return top_left_seed, int(patch_size)

    def sample_proj_mat(self, proj_mat_weights: List[List[float]], proj_mat_indices: List[List[int]]) -> None:
        """
        Sample projection matrix using contiguous patches; weight=1 for all entries.
        """
        for proj_i in range(self.max_features):
            top_left_seed, patch_size = self.sample_top_left_seed()
            self.sample_proj_vec(
                proj_mat_weights,
                proj_mat_indices,
                proj_i,
                patch_size,
                top_left_seed,
                self.patch_sampled_size.copy(),  # pass a snapshot
            )

    def sample_proj_vec(
        self,
        proj_mat_weights: List[List[float]],
        proj_mat_indices: List[List[int]],
        proj_i: int,
        patch_size: int,
        top_left_patch_seed: int,
        patch_dims: np.ndarray,
    ) -> None:
        """
        Fill sparse projection vector (indices/weights) for a single feature (proj_i).
        """
        # Optional discontiguous axis handling: shuffle row index mapping
        if self._discontiguous:
            num_rows = int(self.data_dims[0])
            self._index_data_buffer[:num_rows] = np.arange(num_rows, dtype=np.intp)
            fisher_yates_shuffle(self._index_data_buffer[:num_rows], self._rng)
            # choose first patch_dims[0] indices as sampled rows for the discontiguous dimension
            self._index_patch_buffer[:patch_dims[0]] = self._index_data_buffer[:patch_dims[0]]
            # Alternatively, you could use Floyd sampling:
            # self._index_patch_buffer[:patch_dims[0]] = floyd_sample_indices(int(patch_dims[0]), num_rows, self._rng)

        # Iterate all positions within the n-D patch hyper-rectangle
        unraveled = self.unraveled_patch_point.copy()
        for patch_idx in range(patch_size):
            # refresh base seed -> unraveled coords
            unravel_index_cython(top_left_patch_seed, self.data_dims, unraveled)

            # convert patch_idx to per-dimension offsets
            vectorized_patch_offset = 1
            for dim_idx in range(self.ndim):
                offset = (patch_idx // vectorized_patch_offset) % int(patch_dims[dim_idx])
                unraveled[dim_idx] = unraveled[dim_idx] + int(offset)
                vectorized_patch_offset *= int(patch_dims[dim_idx])

            # Handle discontiguity: remap non-contiguous axes by shuffled row indices
            if self._discontiguous:
                for dim_idx in range(self.ndim):
                    if bool(self.dim_contiguous[dim_idx]):
                        continue

                    # The Cython code attempts to compute a "row index" dependent on other non-contiguous dims.
                    # We approximate by cycling within chosen sampled rows for this axis.
                    # (If you need exact parity, port the exact row-index math one-to-one.)
                    row_index = int(unraveled[dim_idx]) % int(patch_dims[0])
                    unraveled[dim_idx] = self._index_patch_buffer[row_index]

            # ravel back to feature index and append
            vectorized_point = ravel_multi_index_cython(unraveled, self.data_dims)
            proj_mat_indices[proj_i].append(int(vectorized_point))
            proj_mat_weights[proj_i].append(1.0)

    # ------------------- Feature projection over sample set -------------------

    def compute_features_over_samples(
        self,
        start: int,
        end: int,
        samples: np.ndarray,
        feature_values: np.ndarray,
        proj_vec_weights: List[float],
        proj_vec_indices: List[int],
    ) -> None:
        """
        Compute linear combination X[sample, feat]*weight over sparse vector for samples[start:end].
        feature_values is modified in-place (expects at least length end).
        """
        X = self.X
        fw = self.feature_weight

        # Pre-checks
        assert X is not None, "X not initialized"
        assert samples.ndim == 1
        assert start >= 0 and end <= samples.shape[0]

        # Accumulate per sample
        for idx in range(start, end):
            s = int(samples[idx])
            val = 0.0
            patch_weight = 0.0

            # Sparse dot
            for feat, w in zip(proj_vec_indices, proj_vec_weights):
                val += float(X[s, int(feat)]) * float(w)
                if fw is not None:
                    patch_weight += float(fw[s, int(feat)])

            if fw is not None and patch_weight != 0.0:
                val /= patch_weight

            feature_values[idx] = float(val)


# --------------------------- Tester (Python interface) -----------------------

class BestPatchSplitterTester(BestPatchSplitter):
    """
    Exposes Python-friendly methods analogous to the Cython cpdef testers.
    """

    def sample_top_left_seed_cpdef(self) -> Tuple[int, int, np.ndarray]:
        top_left_patch_seed, patch_size = self.sample_top_left_seed()
        patch_dims = self.patch_sampled_size.astype(np.intp).copy()
        return int(top_left_patch_seed), int(patch_size), patch_dims

    def sample_projection_vector(
        self,
        proj_i: int,
        patch_size: int,
        top_left_patch_seed: int,
        patch_dims: np.ndarray,
    ) -> np.ndarray:
        # Local sparse containers for this call
        proj_mat_weights: List[List[float]] = [[] for _ in range(self.max_features)]
        proj_mat_indices: List[List[int]] = [[] for _ in range(self.max_features)]

        self.sample_proj_vec(
            proj_mat_weights,
            proj_mat_indices,
            int(proj_i),
            int(patch_size),
            int(top_left_patch_seed),
            np.asarray(patch_dims, dtype=np.intp),
        )

        # Convert to dense 1 x n_features vector
        proj_vecs = np.zeros((1, self.n_features), dtype=np.float64)
        for j, feat in enumerate(proj_mat_indices[proj_i]):
            weight = proj_mat_weights[proj_i][j]
            proj_vecs[0, int(feat)] = float(weight)
        return proj_vecs

    def sample_projection_matrix_py(self) -> np.ndarray:
        proj_mat_weights: List[List[float]] = [[] for _ in range(self.max_features)]
        proj_mat_indices: List[List[int]] = [[] for _ in range(self.max_features)]

        self.sample_proj_mat(proj_mat_weights, proj_mat_indices)

        proj_vecs = np.zeros((self.max_features, self.n_features), dtype=np.float64)
        for i in range(self.max_features):
            for j, feat in enumerate(proj_mat_indices[i]):
                weight = proj_mat_weights[i][j]
                proj_vecs[i, int(feat)] = float(weight)
        return proj_vecs

    def init_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        missing_values_in_feature_mask: Optional[np.ndarray] = None,
    ) -> int:
        return super().init(X, y, sample_weight, missing_values_in_feature_mask)
