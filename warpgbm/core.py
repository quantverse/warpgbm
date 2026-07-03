import torch
import numpy as np
import pickle
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from warpgbm.cuda import node_kernel
from warpgbm.metrics import rmsle_torch, softmax, log_loss_torch, accuracy_torch
from tqdm import tqdm
from typing import Tuple
from torch import Tensor
import gc

BY_ERA_HISTOGRAM_MIN_ROWS = 3072

class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=56,
        L2_reg=1e-6,
        device="cuda",
        colsample_bytree=1.0,
        random_state=None,
        warm_start=False,
    ):
        # Validate arguments
        self._validate_hyperparams(
            objective=objective,
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            colsample_bytree=colsample_bytree,
        )

        self.objective = objective
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None
        self.base_prediction = None
        self.unique_eras = None
        self.device = device
        self.num_classes = None
        self.classes_ = None
        self.root_gradient_histogram = None
        self.root_hessian_histogram = None
        self.gradients = None
        self.root_node_indices = None
        self.bin_indices = None
        self.Y_gpu = None
        self.num_features = None
        self.num_samples = None
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.best_bin = torch.tensor([-1], dtype=torch.int32, device=self.device)
        self.best_split_idx = torch.empty(1, dtype=torch.int32, device=self.device)
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.forest = [{} for _ in range(self.n_estimators)]
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.warm_start = warm_start
        self.label_encoder = None
        self.feature_importance_ = None
        self.per_era_feature_importance_ = None
        self.era_sorted_for_hist = False
        self.era_row_boundaries = None
        self.max_root_era_count = 0
        self._is_fitted = False
        self._trees_trained = 0  # Track number of trees already trained

    def _ensure_feature_importance_buffers(self):
        target_shape = (self.num_eras, self.num_features)
        current_gpu = getattr(self, "per_era_feature_importance_gpu", None)
        if current_gpu is not None and tuple(current_gpu.shape) == target_shape:
            self.per_era_feature_importance_ = current_gpu.detach().cpu().numpy()
            return

        new_gpu = torch.zeros(target_shape, device=self.device, dtype=torch.float32)

        if current_gpu is not None:
            rows = min(current_gpu.shape[0], target_shape[0])
            cols = min(current_gpu.shape[1], target_shape[1])
            if rows > 0 and cols > 0:
                new_gpu[:rows, :cols] = current_gpu[:rows, :cols]
        elif self.per_era_feature_importance_ is not None:
            source = torch.as_tensor(
                self.per_era_feature_importance_,
                device=self.device,
                dtype=torch.float32,
            )
            rows = min(source.shape[0], target_shape[0])
            cols = min(source.shape[1], target_shape[1])
            if rows > 0 and cols > 0:
                new_gpu[:rows, :cols] = source[:rows, :cols]

        self.per_era_feature_importance_gpu = new_gpu
        self.per_era_feature_importance_ = new_gpu.detach().cpu().numpy()

    def _serialize_tree_for_pickle(self, node):
        if isinstance(node, list):
            return [self._serialize_tree_for_pickle(child) for child in node]
        if not isinstance(node, dict):
            return node

        serializable = {}
        for key, value in node.items():
            if key == "leaf_value" and torch.is_tensor(value):
                serializable[key] = float(value.detach().cpu().item())
            elif isinstance(value, dict) or isinstance(value, list):
                serializable[key] = self._serialize_tree_for_pickle(value)
            else:
                serializable[key] = value
        return serializable

    def _tensor_for_pickle(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        return value

    def _validate_hyperparams(self, **kwargs):
        # Validate objective
        if kwargs["objective"] not in ["regression", "multiclass", "binary"]:
            raise ValueError(
                f"objective must be 'regression', 'binary', or 'multiclass', got {kwargs['objective']}."
            )
        
        # Type checks
        int_params = [
            "num_bins",
            "max_depth",
            "n_estimators",
            "min_child_weight",
            "threads_per_block",
            "rows_per_thread",
        ]
        float_params = [
            "learning_rate",
            "min_split_gain",
            "L2_reg",
            "colsample_bytree",
        ]

        for param in int_params:
            if not isinstance(kwargs[param], int):
                raise TypeError(
                    f"{param} must be an integer, got {type(kwargs[param])}."
                )

        for param in float_params:
            if not isinstance(
                kwargs[param], (float, int)
            ):  # Accept ints as valid floats
                raise TypeError(f"{param} must be a float, got {type(kwargs[param])}.")

        if not (2 <= kwargs["num_bins"] <= 127):
            raise ValueError("num_bins must be between 2 and 127 inclusive.")
        if kwargs["max_depth"] < 1:
            raise ValueError("max_depth must be at least 1.")
        if not (0.0 < kwargs["learning_rate"] <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0].")
        if kwargs["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive.")
        if kwargs["min_child_weight"] < 1:
            raise ValueError("min_child_weight must be a positive integer.")
        if kwargs["min_split_gain"] < 0:
            raise ValueError("min_split_gain must be non-negative.")
        if kwargs["threads_per_block"] <= 0 or kwargs["threads_per_block"] % 32 != 0:
            raise ValueError(
                "threads_per_block should be a positive multiple of 32 (warp size)."
            )
        if not (1 <= kwargs["rows_per_thread"] <= 64):
            raise ValueError(
                "rows_per_thread must be positive between 1 and 64 inclusive."
            )
        if kwargs["L2_reg"] < 0:
            raise ValueError("L2_reg must be non-negative.")
        if kwargs["colsample_bytree"] <= 0 or kwargs["colsample_bytree"] > 1:
            raise ValueError(
                f"Invalid colsample_bytree: {kwargs['colsample_bytree']}. Must be a float value > 0 and <= 1."
            )

    def _compute_tree_predictions(self, tree, bin_indices):
        """
        Compute predictions from a single tree for all samples.
        
        Args:
            tree: Tree dict with structure
            bin_indices: Binned features for samples
        
        Returns:
            torch.Tensor: Predictions for each sample
        """
        num_samples = bin_indices.size(0)
        predictions = torch.zeros(num_samples, device=self.device, dtype=torch.float32)
        
        def traverse(node, sample_mask):
            """Recursively traverse tree and assign leaf values."""
            if "leaf_value" in node:
                # Leaf node: assign value to all samples in this leaf
                predictions[sample_mask] = node["leaf_value"] * self.learning_rate
            else:
                # Split node: route samples left or right
                feature_idx = node["feature"]
                split_bin = node["bin"]
                
                # Samples go left if bin_value <= split_bin
                go_left = bin_indices[sample_mask, feature_idx] <= split_bin
                
                left_mask = sample_mask.clone()
                left_mask[sample_mask] = go_left
                right_mask = sample_mask.clone()
                right_mask[sample_mask] = ~go_left
                
                traverse(node["left"], left_mask)
                traverse(node["right"], right_mask)
        
        # Start with all samples
        all_samples = torch.ones(num_samples, dtype=torch.bool, device=self.device)
        traverse(tree, all_samples)
        
        return predictions
    
    def _compute_softmax_gradients_hessians(self, y_true_encoded):
        """
        Compute gradients and hessians for softmax multiclass classification.
        
        Args:
            y_true_encoded: 1D tensor of encoded class labels [n_samples]
        
        Returns:
            gradients: 2D tensor [n_samples, n_classes]
            hessians: 2D tensor [n_samples, n_classes]
        """
        # Compute probabilities from current predictions using softmax
        probs = softmax(self.gradients, dim=1)  # [n_samples, n_classes]
        
        # Create one-hot encoded labels
        n_samples = y_true_encoded.shape[0]
        y_onehot = torch.zeros(n_samples, self.num_classes, device=self.device)
        y_onehot[torch.arange(n_samples), y_true_encoded.long()] = 1.0
        
        # Gradient: p_k - y_k
        gradients = probs - y_onehot
        
        # Hessian: p_k * (1 - p_k) (diagonal approximation)
        # This treats each class independently which is computationally efficient
        hessians = probs * (1.0 - probs)
        # Clamp hessians to avoid numerical issues
        hessians = torch.clamp(hessians, min=1e-6)
        
        return gradients, hessians

    def validate_fit_params(
        self, X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric
    ):
        # ─── Required: X and y ───
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows. Got {X.shape[0]} and {y.shape[0]}."
            )

        # ─── Optional: era_id ───
        if era_id is not None:
            if not isinstance(era_id, np.ndarray):
                raise TypeError("era_id must be a numpy array.")
            if era_id.ndim != 1:
                raise ValueError(
                    f"era_id must be 1-dimensional, got shape {era_id.shape}"
                )
            if len(era_id) != len(y):
                raise ValueError(
                    f"era_id must have same length as y. Got {len(era_id)} and {len(y)}."
                )

        # ─── Optional: Eval Set ───
        eval_args = [X_eval, y_eval, eval_every_n_trees]
        if any(arg is not None for arg in eval_args):
            # Require all of them
            if X_eval is None or y_eval is None or eval_every_n_trees is None:
                raise ValueError(
                    "If using eval set, X_eval, y_eval, and eval_every_n_trees must all be defined."
                )

            if not isinstance(X_eval, np.ndarray) or not isinstance(y_eval, np.ndarray):
                raise TypeError("X_eval and y_eval must be numpy arrays.")
            if X_eval.ndim != 2:
                raise ValueError(
                    f"X_eval must be 2-dimensional, got shape {X_eval.shape}"
                )
            if y_eval.ndim != 1:
                raise ValueError(
                    f"y_eval must be 1-dimensional, got shape {y_eval.shape}"
                )
            if X_eval.shape[0] != y_eval.shape[0]:
                raise ValueError(
                    f"X_eval and y_eval must have same number of rows. Got {X_eval.shape[0]} and {y_eval.shape[0]}."
                )

            if not isinstance(eval_every_n_trees, int) or eval_every_n_trees <= 0:
                raise ValueError(
                    f"eval_every_n_trees must be a positive integer, got {eval_every_n_trees}."
                )

            if early_stopping_rounds is not None:
                if (
                    not isinstance(early_stopping_rounds, int)
                    or early_stopping_rounds <= 0
                ):
                    raise ValueError(
                        f"early_stopping_rounds must be a positive integer, got {early_stopping_rounds}."
                    )
            else:
                # No early stopping = set to "never trigger"
                early_stopping_rounds = self.n_estimators + 1

            valid_metrics = ["mse", "corr", "rmsle", "logloss", "accuracy"]
            if eval_metric not in valid_metrics:
                raise ValueError(
                    f"Invalid eval_metric: {eval_metric}. Choose from {valid_metrics}."
                )

        return early_stopping_rounds  # May have been defaulted here

    def fit(
        self,
        X,
        y,
        era_id=None,
        X_eval=None,
        y_eval=None,
        eval_every_n_trees=None,
        early_stopping_rounds=None,
        eval_metric = "mse",
    ):
        early_stopping_rounds = self.validate_fit_params(
            X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric
        )

        # Set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.cuda.manual_seed_all(self.random_state)
                # Note: Some CUDA operations may still be non-deterministic
                # For full determinism, set: torch.use_deterministic_algorithms(True)
                # but this may impact performance

        # ─── Warm Start Logic ───
        if not self.warm_start or not self._is_fitted:
            # Fresh start: reset training state
            self._is_fitted = False
            self._trees_trained = 0
            self.forest = [{} for _ in range(self.n_estimators)] if self.objective == "regression" else []
            self.training_loss = []
            self.eval_loss = []
        else:
            # Continuing training: validate that data is compatible
            if X.shape[1] != self.num_features:
                raise ValueError(
                    f"X has {X.shape[1]} features, but model was trained with {self.num_features} features."
                )
            if self._trees_trained >= self.n_estimators:
                # Already have all requested trees
                print(f"Model already has {self._trees_trained} trees (n_estimators={self.n_estimators}). No additional training needed.")
                return self

        if era_id is None:
            era_id = np.ones(X.shape[0], dtype="int32")

        # ─── Handle multiclass vs regression ───
        if self.objective == "multiclass" or self.objective == "binary":
            # Encode labels
            if not self.warm_start or not self._is_fitted:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.classes_ = self.label_encoder.classes_
                self.num_classes = len(self.classes_)
            else:
                # Warm start: use existing label encoder
                y_encoded = self.label_encoder.transform(y)
            
            if self.objective == "binary" and self.num_classes != 2:
                raise ValueError(f"binary objective requires exactly 2 classes, got {self.num_classes}")
            if self.objective == "multiclass" and self.num_classes < 2:
                raise ValueError(f"multiclass objective requires at least 2 classes, got {self.num_classes}")
            
            return self._fit_classification(X, y_encoded, era_id, X_eval, y_eval, 
                                           eval_every_n_trees, early_stopping_rounds, eval_metric)
        else:
            # Regression path (unchanged)
            return self._fit_regression(X, y, era_id, X_eval, y_eval,
                                       eval_every_n_trees, early_stopping_rounds, eval_metric)

    def _setup_sorted_era_histogram_boundaries(self, era_id):
        era_values = np.asarray(era_id)
        self.era_sorted_for_hist = False
        self.era_row_boundaries = None
        self.max_root_era_count = 0

        if era_values.ndim != 1 or era_values.size == 0:
            return
        if not np.all(era_values[1:] >= era_values[:-1]):
            return

        unique_eras = np.unique(era_values)
        boundaries = np.empty(unique_eras.size + 1, dtype=np.int32)
        boundaries[:-1] = np.searchsorted(era_values, unique_eras, side="left")
        boundaries[-1] = era_values.size
        self.era_row_boundaries = torch.as_tensor(
            boundaries,
            device=self.device,
            dtype=torch.int32,
        )
        self.max_root_era_count = int(np.diff(boundaries).max())
        self.era_sorted_for_hist = True

    def _fit_regression(self, X, y, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric):
        """Original regression fitting logic"""
        # Train data preprocessing
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = (
            self.preprocess_gpu_data(X, y, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)
        self._setup_sorted_era_histogram_boundaries(era_id)
        
        # Initialize gradients (predictions)
        if self.warm_start and self._is_fitted and self._trees_trained > 0:
            # Warm start: restore predictions from existing forest
            self.gradients = torch.zeros_like(self.Y_gpu) + self.base_prediction
            # Add predictions from existing trees
            for tree in self.forest[:self._trees_trained]:
                if tree:
                    leaf_updates = self._compute_tree_predictions(tree, self.bin_indices)
                    self.gradients += leaf_updates
        else:
            # Fresh start
            self.gradients = torch.zeros_like(self.Y_gpu)
            self.base_prediction = self.Y_gpu.mean().item()
            self.gradients += self.base_prediction
        
        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        # ─── Optional Eval Set ───
        if X_eval is not None and y_eval is not None:
            self.bin_indices_eval = self.bin_inference_data(X_eval)
            self.Y_gpu_eval = torch.from_numpy(y_eval).to(torch.float32).to(self.device)
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = eval_metric
        else:
            self.bin_indices_eval = None
            self.Y_gpu_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None

        # ─── Grow the forest ───
        with torch.no_grad():
            self.grow_forest()

        del self.bin_indices
        if hasattr(self, "bin_indices_hist"):
            del self.bin_indices_hist
        del self.Y_gpu

        gc.collect()

        return self
    
    def _fit_classification(self, X, y_encoded, era_id, X_eval, y_eval, eval_every_n_trees, early_stopping_rounds, eval_metric):
        """Multiclass classification fitting logic"""
        # Train data preprocessing
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras, _ = (
            self.preprocess_gpu_data(X, y_encoded, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)
        self._setup_sorted_era_histogram_boundaries(era_id)
        
        # Store labels as integers
        self.Y_gpu = torch.from_numpy(y_encoded).to(torch.int32).to(self.device)
        
        # Initialize scores F[i,k] - shape (N, K)
        if self.warm_start and self._is_fitted and self._trees_trained > 0:
            # Warm start: restore predictions from existing forest
            self.gradients = torch.zeros((self.num_samples, self.num_classes), 
                                         dtype=torch.float32, device=self.device)
            # Initialize with class priors
            for k in range(self.num_classes):
                prior_k = (self.Y_gpu == k).float().mean()
                if prior_k > 0:
                    self.gradients[:, k] = torch.log(prior_k + 1e-10)
            
            # Add predictions from existing trees
            for round_trees in self.forest[:self._trees_trained]:
                for class_k, tree in enumerate(round_trees):
                    if tree:
                        leaf_updates = self._compute_tree_predictions(tree, self.bin_indices)
                        self.gradients[:, class_k] += leaf_updates
        else:
            # Fresh start
            self.gradients = torch.zeros((self.num_samples, self.num_classes), 
                                         dtype=torch.float32, device=self.device)
            
            # Initialize with class priors (optional but helps convergence)
            for k in range(self.num_classes):
                prior_k = (self.Y_gpu == k).float().mean()
                if prior_k > 0:
                    self.gradients[:, k] = torch.log(prior_k + 1e-10)
        
        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        # ─── Optional Eval Set ───
        if X_eval is not None and y_eval is not None:
            self.bin_indices_eval = self.bin_inference_data(X_eval)
            # Encode eval labels
            y_eval_encoded = self.label_encoder.transform(y_eval)
            self.Y_gpu_eval = torch.from_numpy(y_eval_encoded).to(torch.int32).to(self.device)
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = eval_metric if eval_metric != "mse" else "logloss"
        else:
            self.bin_indices_eval = None
            self.Y_gpu_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None

        # ─── Grow the forest (K trees per iteration) ───
        with torch.no_grad():
            self.grow_forest_multiclass()

        del self.bin_indices
        if hasattr(self, "bin_indices_hist"):
            del self.bin_indices_hist
        del self.Y_gpu

        gc.collect()

        return self

    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape

            Y_gpu = torch.from_numpy(Y_np).type(torch.float32).to(self.device)

            era_id_gpu = torch.from_numpy(era_id_np).type(torch.int32).to(self.device)

            bin_indices = torch.empty(
                (self.num_samples, self.num_features), dtype=torch.int8, device="cuda"
            )

            is_integer_type = np.issubdtype(X_np.dtype, np.integer)
            max_vals = X_np.max(axis=0)

            if is_integer_type and np.all(max_vals < self.num_bins):
                print("Detected pre-binned integer input — skipping quantile binning.")
                bin_indices = torch.as_tensor(
                    np.ascontiguousarray(X_np), device=self.device, dtype=torch.int8
                ).contiguous()

                # We'll store None or an empty tensor in self.bin_edges
                # to indicate that we skip binning at predict-time
                bin_edges = torch.arange(
                    1, self.num_bins, dtype=torch.float32
                ).repeat(self.num_features, 1)
                bin_edges = bin_edges.to(self.device)
                unique_eras, era_indices = torch.unique(
                    era_id_gpu, return_inverse=True
                )
                return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu
            
            print("quantile binning.")

            bin_edges = torch.empty(
                (self.num_features, self.num_bins - 1),
                dtype=torch.float32,
                device="cuda",
            )

            for f in range(self.num_features):
                X_f = torch.as_tensor( X_np[:, f], device=self.device, dtype=torch.float32 ).contiguous()
                quantiles = torch.linspace(
                    0, 1, self.num_bins + 1, device="cuda", dtype=X_f.dtype
                )[1:-1]
                bin_edges_f = torch.quantile(
                    X_f, quantiles, dim=0
                ).contiguous()  # shape: [B-1] for 1D input
                bin_indices_f = bin_indices[:, f].contiguous()  # view into output
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:, f] = bin_indices_f
                bin_edges[f, :] = bin_edges_f

            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
            return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, sample_indices, feature_indices):
        if (
            getattr(self, "era_sorted_for_hist", False)
            and sample_indices.numel() > BY_ERA_HISTOGRAM_MIN_ROWS
        ):
            return self.compute_histograms_by_era(sample_indices, feature_indices)

        grad_hist = torch.zeros(
            ( self.num_eras, len(feature_indices), self.num_bins), device=self.device, dtype=torch.float32
        )
        hess_hist = torch.zeros(
            ( self.num_eras, len(feature_indices), self.num_bins), device=self.device, dtype=torch.float32
        )

        node_kernel.compute_histogram_feature_tile(
            self.bin_indices,
            self.residual,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist, hess_hist

    def compute_histograms_by_era(self, sample_indices, feature_indices):
        grad_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins),
            device=self.device,
            dtype=torch.float32,
        )
        hess_hist = torch.zeros_like(grad_hist)

        era_offsets = torch.searchsorted(
            sample_indices,
            self.era_row_boundaries,
            out_int32=True,
        )
        max_era_count = min(int(sample_indices.numel()), int(self.max_root_era_count))
        if max_era_count == 0:
            return grad_hist, hess_hist

        node_kernel.compute_histogram_feature_tile_by_era(
            self.bin_indices,
            self.residual,
            sample_indices,
            feature_indices,
            era_offsets,
            grad_hist,
            hess_hist,
            self.num_bins,
            max_era_count,
            0,
            self.num_eras,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist, hess_hist

    def compute_root_gradient_histogram_by_era(self):
        grad_hist = torch.zeros(
            (self.num_eras, self.num_features, self.num_bins),
            device=self.device,
            dtype=torch.float32,
        )
        if self.max_root_era_count == 0:
            return grad_hist

        node_kernel.compute_histogram_feature_tile_by_era_grad(
            self.bin_indices,
            self.residual,
            self.root_node_indices,
            self.feature_indices,
            self.era_row_boundaries,
            grad_hist,
            self.num_bins,
            self.max_root_era_count,
            0,
            self.num_eras,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist

    def find_best_split(self, gradient_histogram, hessian_histogram):
        if self.num_eras > 1:
            return self._find_best_split_fused(gradient_histogram, hessian_histogram)

        # Single-era fast path (unchanged)
        node_kernel.compute_split(
            gradient_histogram,
            hessian_histogram,
            self.min_split_gain,
            self.min_child_weight,
            self.L2_reg,
            self.per_era_gain,
            self.per_era_direction,
            self.threads_per_block
        )

        era_splitting_criterion = self.per_era_gain[0,:,:]  # [F, B-1]
        dir_score_mask = era_splitting_criterion > self.min_split_gain

        if not dir_score_mask.any():
            return -1, -1

        era_splitting_criterion[dir_score_mask == 0] = float("-inf")
        best_idx = torch.argmax(era_splitting_criterion)
        split_bins = self.num_bins - 1
        best_feature = best_idx // split_bins
        best_bin = best_idx % split_bins

        return best_feature.item(), best_bin.item()

    def _find_best_split_fused(self, gradient_histogram, hessian_histogram):
        """Multi-era path: fused kernel reduces across eras on-GPU, outputs [F, B-1]."""
        compute_split = (
            node_kernel.compute_split_fused_no_atomic
            if self.num_bins <= 9
            else node_kernel.compute_split_fused
        )
        compute_split(
            gradient_histogram,
            hessian_histogram,
            self.min_split_gain,
            self.min_child_weight,
            self.L2_reg,
            self.fused_gain,
            self.fused_agreement,
            self.threads_per_block
        )

        node_kernel.select_split_fused(
            self.fused_gain,
            self.fused_agreement,
            self.min_split_gain,
            self.best_split_idx,
            self.threads_per_block
        )
        best_idx = int(self.best_split_idx.item())
        if best_idx < 0:
            return -1, -1

        split_bins = self.num_bins - 1
        best_feature = best_idx // split_bins
        best_bin = best_idx % split_bins

        return best_feature, best_bin


    def _make_leaf(self, node_indices, class_k=None):
        leaf_value = self.residual[node_indices].mean()
        if class_k is not None:
            # Multiclass: update specific class column
            self.gradients[node_indices, class_k] += self.learning_rate * leaf_value
        else:
            # Regression: update 1D gradients
            self.gradients[node_indices] += self.learning_rate * leaf_value
        return {"leaf_value": leaf_value.detach(), "samples": node_indices.numel()}


    def grow_tree(self, gradient_histogram, hessian_histogram, node_indices, depth, class_k=None):
        if depth == self.max_depth:
            return self._make_leaf(node_indices, class_k)

        parent_size = node_indices.numel()
        if class_k is None and parent_size < 2 * self.min_child_weight:
            return self._make_leaf(node_indices, class_k)

        local_feature, best_bin = self.find_best_split(
            gradient_histogram, hessian_histogram
        )

        if local_feature == -1:
            return self._make_leaf(node_indices, class_k)
        
        # Track feature importance for the chosen feature/bin in one CUDA launch.
        if self.colsample_bytree < 1.0:
            split_feature = int(self.feat_indices_tree[local_feature].item())
        else:
            split_feature = local_feature
        node_kernel.accumulate_feature_importance(
            gradient_histogram,
            hessian_histogram,
            local_feature,
            best_bin,
            split_feature,
            self.min_child_weight,
            self.L2_reg,
            self.per_era_feature_importance_gpu,
            self.threads_per_block,
        )
        
        split_mask = self.bin_indices[node_indices, split_feature] <= best_bin
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        left_size = left_indices.numel()
        right_size = right_indices.numel()
        new_depth = depth + 1

        if new_depth == self.max_depth:
            left_child = self._make_leaf(left_indices, class_k)
            right_child = self._make_leaf(right_indices, class_k)
            return {
                "feature": split_feature,
                "bin": best_bin,
                "left": left_child,
                "right": right_child,
            }

        if left_size <= right_size:
            grad_hist_left, hess_hist_left = self.compute_histograms( left_indices, self.feat_indices_tree )
            node_kernel.subtract_histograms_inplace(
                gradient_histogram,
                hessian_histogram,
                grad_hist_left,
                hess_hist_left,
                256,
            )
            grad_hist_right = gradient_histogram
            hess_hist_right = hessian_histogram
        else:
            grad_hist_right, hess_hist_right = self.compute_histograms( right_indices, self.feat_indices_tree )
            node_kernel.subtract_histograms_inplace(
                gradient_histogram,
                hessian_histogram,
                grad_hist_right,
                hess_hist_right,
                256,
            )
            grad_hist_left = gradient_histogram
            hess_hist_left = hessian_histogram

        left_child = self.grow_tree(
            grad_hist_left, hess_hist_left, left_indices, new_depth, class_k
        )
        right_child = self.grow_tree(
            grad_hist_right, hess_hist_right, right_indices, new_depth, class_k
        )

        return {
            "feature": split_feature,
            "bin": best_bin,
            "left": left_child,
            "right": right_child,
        }
    
    def get_eval_metric(self, y_true, y_pred):
        if self.eval_metric == "mse":
            return ((y_true - y_pred) ** 2).mean().item()
        elif self.eval_metric == "corr":
            return 1 - torch.corrcoef(torch.vstack([y_true, y_pred]))[0, 1].item()
        elif self.eval_metric == "rmsle":
            return rmsle_torch(y_true, y_pred).item()
        else:
            raise ValueError(f"Invalid eval_metric: {self.eval_metric}.")

    def compute_eval(self, i):
        if self.eval_every_n_trees == None:
            return
        
        train_loss = ((self.Y_gpu - self.gradients) ** 2).mean().item()
        self.training_loss.append(train_loss)

        if i % self.eval_every_n_trees == 0:
            eval_preds = self.predict_binned(self.bin_indices_eval)
            eval_loss = self.get_eval_metric( self.Y_gpu_eval, eval_preds )
            self.eval_loss.append(eval_loss)

            if len(self.eval_loss) > self.early_stopping_rounds:
                if self.eval_loss[-(self.early_stopping_rounds+1)] < self.eval_loss[-1]:
                    self.stop = True

            print(
                f"🌲 Tree {i+1}/{self.n_estimators} | Train MSE: {train_loss:.6f} | Eval {self.eval_metric}: {eval_loss:.6f}"
            )

            del eval_preds, eval_loss, train_loss

    def grow_forest(self):
        """Regression forest growing (original logic)"""
        # Warm start: preserve existing training state
        if not hasattr(self, 'training_loss') or not self.warm_start or not self._is_fitted:
            self.training_loss = []
            self.eval_loss = []  # if eval set is given
            self.per_era_feature_importance_ = np.zeros((self.num_eras, self.num_features), dtype=np.float32)
            self.per_era_feature_importance_gpu = torch.zeros(
                (self.num_eras, self.num_features),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._ensure_feature_importance_buffers()
        
        self.stop = False

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features
            
        if self.num_eras > 1:
            self.fused_gain = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.fused_agreement = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
        else:
            self.per_era_gain = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.per_era_direction = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)

        # Warm start: start from where we left off
        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0
        can_reuse_root_hessian = (
            self.era_sorted_for_hist
            and self.colsample_bytree >= 1.0
            and self.max_root_era_count > 0
        )
        root_hessian_histogram_cache = None

        # Ensure forest has enough slots
        if len(self.forest) < self.n_estimators:
            self.forest.extend([{} for _ in range(self.n_estimators - len(self.forest))])

        for i in range(start_iter, self.n_estimators):
            self.residual = self.Y_gpu - self.gradients

            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]

            if can_reuse_root_hessian and root_hessian_histogram_cache is not None:
                self.root_gradient_histogram = self.compute_root_gradient_histogram_by_era()
                self.root_hessian_histogram = root_hessian_histogram_cache.clone()
            else:
                self.root_gradient_histogram, self.root_hessian_histogram = self.compute_histograms(
                    self.root_node_indices, self.feat_indices_tree
                )
                if can_reuse_root_hessian:
                    root_hessian_histogram_cache = self.root_hessian_histogram.clone()

            tree = self.grow_tree(
                self.root_gradient_histogram,
                self.root_hessian_histogram,
                self.root_node_indices,
                0,
            )
            self.forest[i] = tree
            self._trees_trained = i + 1

            self.compute_eval(i)

            if self.stop:
                break

        # Aggregate feature importance across eras
        self.per_era_feature_importance_ = (
            self.per_era_feature_importance_gpu.detach().cpu().numpy()
        )
        self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0)
        self._is_fitted = True
        
        print(f"Finished training forest. Total trees: {self._trees_trained}")

    def grow_forest_multiclass(self):
        """Multiclass forest growing - K trees per iteration"""
        # Warm start: preserve existing training state
        if not hasattr(self, 'training_loss') or not self.warm_start or not self._is_fitted:
            self.training_loss = []
            self.eval_loss = []
            self.per_era_feature_importance_ = np.zeros((self.num_eras, self.num_features), dtype=np.float32)
            self.per_era_feature_importance_gpu = torch.zeros(
                (self.num_eras, self.num_features),
                device=self.device,
                dtype=torch.float32,
            )
            # Store K trees per iteration
            self.forest = []
        else:
            self._ensure_feature_importance_buffers()
        
        self.stop = False

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features
            
        if self.num_eras > 1:
            self.fused_gain = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.fused_agreement = torch.zeros(k, self.num_bins-1, device=self.device, dtype=torch.float32)
        else:
            self.per_era_gain = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)
            self.per_era_direction = torch.zeros(1, k, self.num_bins-1, device=self.device, dtype=torch.float32)

        # Warm start: start from where we left off
        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0

        for i in range(start_iter, self.n_estimators):
            # Compute softmax probabilities and gradients/hessians for all classes
            grads, hess = self._compute_softmax_gradients_hessians(self.Y_gpu)
            
            # Train K trees (one per class)
            trees_k = []
            
            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]
            
            for class_k in range(self.num_classes):
                # Set residual for this class (negative gradient)
                self.residual = -grads[:, class_k]
                
                # Compute histograms for this class
                # For multiclass, we treat hessian as "weights" - use them directly
                # We'll compute histograms by accumulating grad and hess
                self.root_gradient_histogram, self.root_hessian_histogram = (
                    self.compute_histograms_multiclass(
                        self.root_node_indices, 
                        self.feat_indices_tree, 
                        self.residual,
                        hess[:, class_k]
                    )
                )
                
                # Grow tree for this class (pass class_k to update correct column)
                tree_k = self.grow_tree(
                    self.root_gradient_histogram,
                    self.root_hessian_histogram,
                    self.root_node_indices,
                    0,
                    class_k=class_k,
                )
                trees_k.append(tree_k)
            
            self.forest.append(trees_k)
            self._trees_trained = i + 1
            
            self.compute_eval_multiclass(i)

            if self.stop:
                break

        # Aggregate feature importance across eras
        self.per_era_feature_importance_ = (
            self.per_era_feature_importance_gpu.detach().cpu().numpy()
        )
        self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0)
        self._is_fitted = True
        
        print(f"Finished training multiclass forest. Total rounds: {self._trees_trained} ({self._trees_trained * self.num_classes} trees)")
    
    def compute_eval_multiclass(self, i):
        """Evaluation for multiclass"""
        if self.eval_every_n_trees is None:
            return
        
        # Compute training loss
        probs_train = softmax(self.gradients, dim=1)
        train_loss = log_loss_torch(self.Y_gpu, probs_train).item()
        self.training_loss.append(train_loss)

        if i % self.eval_every_n_trees == 0:
            # Get predictions on eval set
            eval_probs = self.predict_proba_binned(self.bin_indices_eval)
            
            if self.eval_metric == "logloss":
                eval_loss = log_loss_torch(self.Y_gpu_eval, eval_probs).item()
            elif self.eval_metric == "accuracy":
                eval_preds = torch.argmax(eval_probs, dim=1)
                eval_loss = 1.0 - accuracy_torch(self.Y_gpu_eval, eval_preds).item()
            else:
                eval_loss = log_loss_torch(self.Y_gpu_eval, eval_probs).item()
            
            self.eval_loss.append(eval_loss)

            if len(self.eval_loss) > self.early_stopping_rounds:
                if self.eval_loss[-(self.early_stopping_rounds+1)] < self.eval_loss[-1]:
                    self.stop = True

            print(
                f"🌲 Round {i+1}/{self.n_estimators} ({self.num_classes} trees) | "
                f"Train LogLoss: {train_loss:.6f} | Eval {self.eval_metric}: {eval_loss:.6f}"
            )

            del eval_probs, eval_loss, train_loss
    
    def compute_histograms_multiclass(self, sample_indices, feature_indices, grad, hess):
        """
        Compute histograms for multiclass - similar to regression but with explicit grad/hess
        
        Args:
            sample_indices: which samples to include
            feature_indices: which features to use  
            grad: gradient for this class [n_samples]
            hess: hessian for this class [n_samples]
        """
        grad_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins), 
            device=self.device, dtype=torch.float32
        )
        hess_hist = torch.zeros(
            (self.num_eras, len(feature_indices), self.num_bins), 
            device=self.device, dtype=torch.float32
        )

        # We need to create temporary tensors that match what compute_histogram3 expects
        # It expects self.residual, but we have grad/hess per class
        # Save the original residual
        saved_residual = self.residual if hasattr(self, 'residual') else None
        
        # Create a dummy dataset structure for histogram computation
        # The histogram kernel accumulates residuals, so we pass grad directly as residual
        self.residual = grad
        
        node_kernel.compute_histogram3(
            self.bin_indices,
            self.residual,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        
        # Restore
        if saved_residual is not None:
            self.residual = saved_residual
            
        return grad_hist, hess_hist

    def bin_data_with_existing_edges(self, X_np):
        num_samples = X_np.shape[0]
        bin_indices = torch.zeros(
            (num_samples, self.num_features), dtype=torch.int8, device=self.device
        )
        with torch.no_grad():
            for f in range(self.num_features):
                X_f = torch.as_tensor( X_np[:, f], device=self.device, dtype=torch.float32 ).contiguous()
                bin_edges_f = self.bin_edges[f]
                bin_indices_f = bin_indices[:, f].contiguous()
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:, f] = bin_indices_f

        return bin_indices

    def predict_binned(self, bin_indices):
        num_samples = bin_indices.size(0)
        tree_tensor = torch.stack(
            [
                self.flatten_tree(tree, max_nodes=2 ** (self.max_depth + 1))
                for tree in self.forest
                if tree
            ]
        ).to(self.device)

        out = torch.zeros(num_samples, device=self.device) + self.base_prediction
        node_kernel.predict_forest(
            bin_indices.contiguous(), tree_tensor.contiguous(), self.learning_rate, out
        )

        return out
    
    def bin_inference_data(self, X_np):
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)

        if is_integer_type and X_np.shape[1] == self.num_features:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                print("Detected pre-binned input at predict-time — skipping binning.")
                is_prebinned = True
            else:
                is_prebinned = False
        else:
            is_prebinned = False

        if is_prebinned:
            bin_indices = torch.empty(
                X_np.shape, dtype=torch.int8, device="cuda"
            )
            for f in range(self.num_features):
                bin_indices[:,f] = torch.as_tensor( X_np[:, f], device=self.device).contiguous()
        else:
            bin_indices = self.bin_data_with_existing_edges(X_np)
        return bin_indices

    def predict(self, X_np):
        """
        Predict on new data.
        
        For regression: returns predicted values
        For classification: returns predicted class labels
        """
        if self.objective == "multiclass" or self.objective == "binary":
            # Classification: return class labels
            probs = self.predict_proba(X_np)
            class_indices = np.argmax(probs, axis=1)
            return self.label_encoder.inverse_transform(class_indices)
        else:
            # Regression: return values
            bin_indices = self.bin_inference_data(X_np)
            preds = self.predict_binned(bin_indices).cpu().numpy()
            del bin_indices
            return preds
    
    def predict_proba(self, X_np):
        """
        Predict class probabilities (classification only).
        
        Returns:
            np.array of shape (n_samples, n_classes) with probabilities
        """
        if self.objective not in ["multiclass", "binary"]:
            raise ValueError("predict_proba only available for classification objectives")
        
        bin_indices = self.bin_inference_data(X_np)
        probs = self.predict_proba_binned(bin_indices).cpu().numpy()
        del bin_indices
        return probs
    
    def predict_proba_binned(self, bin_indices):
        """
        Predict probabilities from pre-binned data (multiclass).
        
        Returns:
            torch.Tensor of shape (n_samples, n_classes) with probabilities
        """
        num_samples = bin_indices.size(0)
        
        # Initialize raw scores (logits) for all classes
        F = torch.zeros((num_samples, self.num_classes), device=self.device, dtype=torch.float32)
        
        # Initialize with class priors if they were used during training
        for k in range(self.num_classes):
            prior_k = (self.classes_ == self.classes_[k]).sum() / len(self.classes_)
            if prior_k > 0:
                F[:, k] = torch.log(torch.tensor(prior_k + 1e-10))
        
        # Accumulate predictions from all trees
        for round_trees in self.forest:
            if not round_trees:  # Skip if empty
                continue
            for k, tree in enumerate(round_trees):
                # Get predictions for class k
                tree_tensor = self.flatten_tree(tree, max_nodes=2 ** (self.max_depth + 1)).to(self.device)
                tree_preds = torch.zeros(num_samples, device=self.device)
                
                # Call predict kernel for this single tree
                node_kernel.predict_forest(
                    bin_indices.contiguous(),
                    tree_tensor.unsqueeze(0).contiguous(),  # Add batch dimension
                    self.learning_rate,
                    tree_preds
                )
                
                F[:, k] += tree_preds
        
        # Convert logits to probabilities via softmax
        probs = softmax(F, dim=1)
        return probs

    def get_feature_importance(self, importance_type='gain', normalize=True):
        """
        Get feature importance scores.
        
        Parameters:
        -----------
        importance_type : str, default='gain'
            Type of importance. Currently only 'gain' is supported.
        normalize : bool, default=True
            If True, normalize importances to sum to 1.0
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_features,) with importance scores.
            
        Note:
        -----
        When era_id is used during training, this returns the sum of 
        per-era importances. Use get_per_era_feature_importance() to 
        see era-specific importances.
        """
        if self.feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        if importance_type != 'gain':
            raise ValueError(f"importance_type '{importance_type}' not supported. Use 'gain'.")
        
        importance = self.feature_importance_.copy()
        
        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()
        
        return importance
    
    def get_per_era_feature_importance(self, normalize=True):
        """
        Get per-era feature importance scores.
        
        Parameters:
        -----------
        normalize : bool, default=True
            If True, normalize importances within each era to sum to 1.0
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_eras, n_features) with importance scores per era.
            
        Note:
        -----
        This is unique to WarpGBM's era-splitting capability. When no era_id
        is provided during training (standard ERM setting), n_eras=1 and this
        returns the same as get_feature_importance() but with shape (1, n_features).
        
        This allows you to see which features are invariant across eras vs
        which features are only important in specific eras.
        """
        if self.per_era_feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        importance = self.per_era_feature_importance_.copy()
        
        if normalize:
            for era_idx in range(importance.shape[0]):
                era_sum = importance[era_idx].sum()
                if era_sum > 0:
                    importance[era_idx] /= era_sum
        
        return importance

    def save_model(self, path):
        """
        Save the trained model to disk for later use.
        
        Parameters:
        -----------
        path : str
            File path where the model will be saved (e.g., 'model.pkl')
        
        Example:
        --------
        >>> model = WarpGBM(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> model.save_model('checkpoint_100.pkl')
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        # Collect all the necessary state to save
        state = {
            # Hyperparameters
            'objective': self.objective,
            'num_bins': self.num_bins,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_weight': self.min_child_weight,
            'min_split_gain': self.min_split_gain,
            'L2_reg': self.L2_reg,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            
            # Trained state
            'forest': self._serialize_tree_for_pickle(self.forest),
            'bin_edges': self._tensor_for_pickle(self.bin_edges),
            'base_prediction': self.base_prediction,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'classes_': self.classes_,
            'label_encoder': self.label_encoder,
            'feature_importance_': self.feature_importance_,
            'per_era_feature_importance_': self.per_era_feature_importance_,
            '_is_fitted': self._is_fitted,
            '_trees_trained': self._trees_trained,
            'training_loss': self.training_loss if hasattr(self, 'training_loss') else [],
            'eval_loss': self.eval_loss if hasattr(self, 'eval_loss') else [],
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a previously saved model from disk.
        
        Parameters:
        -----------
        path : str
            File path to the saved model (e.g., 'model.pkl')
        
        Returns:
        --------
        self : WarpGBM
            The model instance with loaded state
        
        Example:
        --------
        >>> model = WarpGBM()
        >>> model.load_model('checkpoint_100.pkl')
        >>> # Continue training with warm_start
        >>> model.warm_start = True
        >>> model.n_estimators = 200  # Train 100 more trees
        >>> model.fit(X_train, y_train)
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Restore hyperparameters
        self.objective = state['objective']
        self.num_bins = state['num_bins']
        self.max_depth = state['max_depth']
        self.learning_rate = state['learning_rate']
        self.n_estimators = state['n_estimators']
        self.min_child_weight = state['min_child_weight']
        self.min_split_gain = state['min_split_gain']
        self.L2_reg = state['L2_reg']
        self.colsample_bytree = state['colsample_bytree']
        self.random_state = state['random_state']
        self.warm_start = state['warm_start']
        
        # Restore trained state
        self.forest = state['forest']
        self.bin_edges = state['bin_edges']
        if torch.is_tensor(self.bin_edges) and self.device != "cpu" and torch.cuda.is_available():
            self.bin_edges = self.bin_edges.to(self.device)
        self.base_prediction = state['base_prediction']
        self.num_features = state['num_features']
        self.num_classes = state['num_classes']
        self.classes_ = state['classes_']
        self.label_encoder = state['label_encoder']
        self.feature_importance_ = state['feature_importance_']
        self.per_era_feature_importance_ = state['per_era_feature_importance_']
        self._is_fitted = state['_is_fitted']
        self._trees_trained = state['_trees_trained']
        self.training_loss = state.get('training_loss', [])
        self.eval_loss = state.get('eval_loss', [])
        
        print(f"Model loaded from {path} ({self._trees_trained} trees)")
        
        return self

    def flatten_tree(self, tree, max_nodes):
        flat = torch.full((max_nodes, 6), float("nan"), dtype=torch.float32)
        node_counter = [0]
        node_list = []

        def walk(node):
            curr_id = node_counter[0]
            node_counter[0] += 1

            new_node = {"node_id": curr_id}
            if "leaf_value" in node:
                new_node["leaf_value"] = float(node["leaf_value"])
            else:
                new_node["best_feature"] = float(node["feature"])
                new_node["split_bin"] = float(node["bin"])
                new_node["left_id"] = node_counter[0]
                walk(node["left"])
                new_node["right_id"] = node_counter[0]
                walk(node["right"])

            node_list.append(new_node)
            return new_node

        walk(tree)

        for node in node_list:
            i = node["node_id"]
            if "leaf_value" in node:
                flat[i, 4] = 1.0
                flat[i, 5] = node["leaf_value"]
            else:
                flat[i, 0] = node["best_feature"]
                flat[i, 1] = node["split_bin"]
                flat[i, 2] = node["left_id"]
                flat[i, 3] = node["right_id"]
                flat[i, 4] = 0.0

        return flat
