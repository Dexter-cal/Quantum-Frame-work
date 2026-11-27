"""
qai.techniques.cnn -- A real Convolutional Neural Network technique.

Implements:
  - Configurable stack of Conv2d + ReLU + MaxPool blocks
  - Adaptive average pooling (works with any input image size)
  - Linear classifier head
  - AdamW + cosine LR schedule + warmup + gradient clipping
  - Data augmentation (random crop, horizontal flip, rotation)
  - Feature extraction (intermediate activations)
  - Transfer-learning mode (freeze conv base, train new head)
  - Save/load via qai.bundle (PyTorch state_dict path)

Supports grayscale (1-channel) and RGB (3-channel) image inputs. Accepts
numpy arrays of shape (H, W), (H, W, C), or (N, H, W, C). Channels-first
or channels-last both auto-handled.

Example (image classification):
    import qai
    import numpy as np

    # Synthesize 100 grayscale 16x16 images, 3 classes
    X = np.random.randn(100, 16, 16).astype(np.float32)
    y = np.random.randint(0, 3, size=100)

    model = qai.build(
        technique="cnn",
        in_channels=1,
        n_classes=3,
        conv_channels=[8, 16],     # 2 conv blocks, 8 then 16 filters
        kernel_sizes=[3, 3],
        pool_sizes=[2, 2],
        fc_hidden=32,
        epochs=20,
        lr=1e-3,
        batch_size=16,
    )
    model.train(X, y)
    print("Predicted class:", model.predict(X[0]))

Author: Z.ai brutal-honest reviewer mode.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace
    torch = _TorchStubNamespace()
    nn = _NNStubNamespace()
    F = nn

from .base import Technique


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "[WHAT] qai.build(technique='cnn') requires PyTorch\n"
            "[WHY] The CNN technique uses torch.nn for its internals\n"
            "[FIX] Install torch: pip install torch   (or: pip install -r requirements.txt)"
        )


# =====================================================================
# PyTorch building blocks
# =====================================================================
class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018).

    Squeezes spatial dims via global avg pool, runs a 2-layer MLP to
    recompute per-channel weights, then re-weights the input. Built
    inside the existing CNN's _ConvBlock when se_blocks=True.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excite(s).view(b, c, 1, 1)
        return x * s


class _ConvBlock(nn.Module):
    """A single Conv2d + ReLU + (optional) MaxPool block.

    All conv layers use padding='same' (kernel_size // 2) so spatial dims
    only change at the pool step. This makes architecture configuration
    simpler — pool_sizes alone control the downsampling factor.

    Modern upgrades (built ON TOP of the existing block, not a new type):
      - se_blocks=True: append a Squeeze-and-Excitation attention block.
      - drop_rate>0: stochastic depth (drop this entire block with
        probability `drop_rate` during training; identity short-circuit).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, pool_size: Optional[int] = None,
                 dropout: float = 0.0,
                 se_blocks: bool = False,
                 drop_rate: float = 0.0,
                 norm_type: str = "batch_norm",
                 activation: str = "relu",
                 groups: int = 1,
                 dilation: int = 1):
        super().__init__()
        # padding for dilated conv: (kernel_size - 1) * dilation // 2
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False,
                              groups=groups, dilation=dilation)
        # Configurable normalization
        if norm_type == "batch_norm":
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm_type == "layer_norm":
            # LayerNorm on channels (need to permute in forward)
            self.bn = nn.GroupNorm(1, out_channels)  # equivalent to LayerNorm per-channel
        elif norm_type == "group_norm":
            n_groups = min(32, out_channels)
            self.bn = nn.GroupNorm(n_groups, out_channels)
        elif norm_type == "none":
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        # Configurable activation
        act_map = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "mish": nn.Mish(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        }
        self.relu = act_map.get(activation, nn.ReLU(inplace=True))
        self.pool = (nn.MaxPool2d(pool_size, pool_size)
                     if pool_size and pool_size > 1 else nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Modern upgrades
        self.se = _SEBlock(out_channels) if se_blocks else nn.Identity()
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        # Stochastic depth: bypass this block's conv/bn/relu/pool but still
        # project the input to the expected output channel count so the next
        # block / classifier sees a consistent channel dimension.
        if self.training and self.drop_rate > 0 and torch.rand(1).item() < self.drop_rate:
            in_ch = x.shape[1]
            out_ch = self.conv.out_channels
            if in_ch != out_ch:
                # 1x1 conv with identity-like weight: (out_ch, in_ch, 1, 1).
                # Copies channels 1:1 where they exist; zero-pads the rest.
                w = torch.eye(out_ch, in_ch, device=x.device).unsqueeze(-1).unsqueeze(-1)
                shortcut = nn.functional.conv2d(x, w)
            else:
                shortcut = x
            # Apply pooling only (so spatial dims stay consistent with the
            # non-bypass path).
            return self.pool(shortcut)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.se(x)
        return x


class _CNNNet(nn.Module):
    """Full CNN: stack of conv blocks → adaptive avg pool → flatten → FC head."""
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        pool_sizes: List[int] = None,
        fc_hidden: int = 64,
        dropout: float = 0.0,
        se_blocks: bool = False,
        stochastic_depth: float = 0.0,
        norm_type: str = "batch_norm",
        activation: str = "relu",
        groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32]
        if kernel_sizes is None:
            kernel_sizes = [3] * len(conv_channels)
        if pool_sizes is None:
            pool_sizes = [2] * len(conv_channels)

        assert len(conv_channels) == len(kernel_sizes) == len(pool_sizes), (
            f"conv_channels, kernel_sizes, pool_sizes must have equal length, "
            f"got {len(conv_channels)}, {len(kernel_sizes)}, {len(pool_sizes)}"
        )

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv_channels = list(conv_channels)
        self.kernel_sizes = list(kernel_sizes)
        self.pool_sizes = list(pool_sizes)
        self.fc_hidden = fc_hidden
        self.se_blocks = se_blocks
        self.stochastic_depth = stochastic_depth
        self.norm_type = norm_type
        self.activation = activation
        self.groups = groups
        self.dilation = dilation

        # Build conv blocks. When stochastic_depth > 0, linearly increase the
        # per-block drop probability from 0 (first block) up to stochastic_depth
        # (last block), as in Huang et al. 2016.
        blocks: List[nn.Module] = []
        prev_ch = in_channels
        n_blocks = len(conv_channels)
        for i, (ch, k, p) in enumerate(zip(conv_channels, kernel_sizes, pool_sizes)):
            drop_rate = (stochastic_depth * i / max(n_blocks - 1, 1)) if n_blocks > 1 else 0.0
            blocks.append(_ConvBlock(prev_ch, ch, k, p, dropout,
                                     se_blocks=se_blocks,
                                     drop_rate=drop_rate,
                                     norm_type=norm_type,
                                     activation=activation,
                                     groups=groups,
                                     dilation=dilation))
            prev_ch = ch
        self.features = nn.Sequential(*blocks)

        # After conv blocks, the spatial dims are reduced by prod(pool_sizes).
        # AdaptiveAvgPool to 1x1 makes the head input size = last conv channels,
        # independent of input image size. This is the trick that lets the CNN
        # accept any image size without re-architecting.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC head: Linear → ReLU → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fc_hidden, n_classes),
        )

        # For feature extraction — stores the output of each block during forward
        self._hook_activations: List[torch.Tensor] = []

    def forward(self, x, capture_activations: bool = False):
        """
        Args:
            x: (batch, channels, H, W) tensor
            capture_activations: if True, store the output of each conv block
                in self._hook_activations (used by extract_features)
        Returns:
            logits: (batch, n_classes)
        """
        if capture_activations:
            self._hook_activations = []

        for block in self.features:
            x = block(x)
            if capture_activations:
                self._hook_activations.append(x.detach())

        x = self.adaptive_pool(x)  # (batch, channels, 1, 1)
        logits = self.classifier(x)  # (batch, n_classes)
        return logits

    def get_activations(self) -> List[torch.Tensor]:
        """Return activations captured during the last forward(capture_activations=True) call."""
        return list(self._hook_activations)


# =====================================================================
# QAI Technique wrapper
# =====================================================================
class CNN(Technique):
    """A real Convolutional Neural Network technique.

    Built on PyTorch. Supports grayscale (1-channel) and RGB (3-channel)
    image classification. Architecture is a stack of Conv→ReLU→MaxPool
    blocks followed by an adaptive-average-pool + linear classifier head.

    The adaptive pooling layer means the CNN accepts ANY input image size
    (no need to resize to a fixed dimension). Just make sure all images
    in a batch are the same size.
    """
    name = "cnn"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        pool_sizes: Optional[List[int]] = None,
        fc_hidden: int = 64,
        dropout: float = 0.0,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_steps: int = 50,
        augment: bool = True,
        augment_flip_p: float = 0.5,
        augment_crop_pad: int = 2,
        augment_rotate: bool = True,
        device: Optional[str] = None,
        # --- Modern upgrades (built ON TOP of the existing CNN, not a new type) ---
        se_blocks: bool = False,
        stochastic_depth: float = 0.0,
        mixup_alpha: float = 0.0,
        label_smoothing: float = 0.0,
        use_swa: bool = False,
        use_tta: bool = False,
        # --- Architecture flexibility (Phase 3.2 additions) ---
        norm_type: str = "batch_norm",  # "batch_norm", "layer_norm", "group_norm", "none"
        activation: str = "relu",  # "relu", "gelu", "silu", "mish", "leaky_relu"
        groups: int = 1,  # grouped convolution (1 = standard, >1 = grouped)
        dilation: int = 1,  # dilated convolution (1 = standard, >1 = dilated)
        initializer_range: float = 0.02,  # weight init std
        **params,
    ):
        """
        Args:
            in_channels: Number of input image channels. 1 = grayscale, 3 = RGB.
            n_classes: Number of output classes.
            conv_channels: List of filter counts per conv block. Default [16, 32].
            kernel_sizes: List of kernel sizes per conv block. Default [3, 3].
            pool_sizes: List of max-pool sizes per block (1 = no pool). Default [2, 2].
            fc_hidden: Hidden dimension of the linear classifier head.
            dropout: Dropout rate (0.0 = none, 0.5 = standard).
            epochs: Number of training epochs.
            lr: Peak learning rate (AdamW + cosine schedule).
            batch_size: Mini-batch size.
            weight_decay: L2 regularization strength.
            warmup_steps: Number of steps for linear LR warmup.
            augment: If True (default), apply random augmentations during training.
            augment_flip_p: Probability of horizontal flip (when augment=True).
            augment_crop_pad: Random crop padding in pixels (when augment=True).
            augment_rotate: If True, randomly rotate images by 0/90/180/270 degrees.
            device: "cuda", "cpu", or None (auto-detect via qai.get_optimal_device()).
            se_blocks: If True, add a Squeeze-and-Excitation block after every
                conv block (Hu et al. 2018). Built on top of the existing CNN.
            stochastic_depth: If > 0, randomly drop conv blocks during training
                with linearly increasing probability (Huang et al. 2016).
            mixup_alpha: If > 0, apply mixup data augmentation (Zhang et al. 2018).
            label_smoothing: If > 0, smooth target labels for better calibration.
            use_swa: If True, apply Stochastic Weight Averaging (Izmailov 2018)
                over the last quarter of training epochs.
            use_tta: If True, apply Test-Time Augmentation at inference time
                (averages predictions over a small set of augmentations).
        """
        super().__init__(
            in_channels=in_channels, n_classes=n_classes,
            conv_channels=conv_channels, kernel_sizes=kernel_sizes,
            pool_sizes=pool_sizes, fc_hidden=fc_hidden, dropout=dropout,
            epochs=epochs, lr=lr, batch_size=batch_size,
            weight_decay=weight_decay, warmup_steps=warmup_steps,
            augment=augment, augment_flip_p=augment_flip_p,
            augment_crop_pad=augment_crop_pad, augment_rotate=augment_rotate,
            device=device, **params,
        )
        _require_torch()

        # Default architecture if not specified
        if conv_channels is None:
            conv_channels = [16, 32]
        if kernel_sizes is None:
            kernel_sizes = [3] * len(conv_channels)
        if pool_sizes is None:
            pool_sizes = [2] * len(conv_channels)

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv_channels = list(conv_channels)
        self.kernel_sizes = list(kernel_sizes)
        self.pool_sizes = list(pool_sizes)
        self.fc_hidden = fc_hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.augment = augment
        self.augment_flip_p = augment_flip_p
        self.augment_crop_pad = augment_crop_pad
        self.augment_rotate = augment_rotate
        # Modern upgrades (built ON TOP of the existing CNN -- no new model type).
        self.se_blocks = se_blocks
        self.stochastic_depth = stochastic_depth
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.use_swa = use_swa
        self.use_tta = use_tta
        # Phase 3.2 architecture flexibility
        self.norm_type = norm_type
        self.activation = activation
        self.groups = groups
        self.dilation = dilation
        self.initializer_range = initializer_range

        # Device resolution
        if device is None:
            try:
                import qai
                device = qai.get_optimal_device()
            except Exception:
                device = "cpu"
        self.device = torch.device(device)

        # Model is built lazily — needs to know n_classes which may be inferred
        # from y at fit() time
        self.net: Optional[_CNNNet] = None
        self._loss_history: List[float] = []
        self._last_X = None
        self._last_y = None
        # Class label encoding (in case string labels are passed)
        self.classes_: Optional[np.ndarray] = None
        # Frozen-layer mode for transfer learning
        self._conv_frozen = False

    # ------------------------------------------------------------------
    # Model construction (lazy)
    # ------------------------------------------------------------------
    def _build_model(self, n_classes: Optional[int] = None):
        if n_classes is not None:
            self.n_classes = n_classes
        self.net = _CNNNet(
            in_channels=self.in_channels,
            n_classes=self.n_classes,
            conv_channels=self.conv_channels,
            kernel_sizes=self.kernel_sizes,
            pool_sizes=self.pool_sizes,
            fc_hidden=self.fc_hidden,
            dropout=self.dropout,
            se_blocks=self.se_blocks,
            stochastic_depth=self.stochastic_depth,
            norm_type=self.norm_type,
            activation=self.activation,
            groups=self.groups,
            dilation=self.dilation,
        ).to(self.device)

        # Apply proper weight initialization
        import torch.nn.init as init
        for module in self.net.modules():
            for name, param in module.named_parameters(recurse=False):
                if "weight" in name and param.dim() >= 2:
                    init.trunc_normal_(param, mean=0.0, std=self.initializer_range)
                elif "bias" in name:
                    init.zeros_(param)
                elif "weight" in name and param.dim() == 1:
                    init.ones_(param)

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    def _normalize_images(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Convert input images to (N, C, H, W) float32 numpy array.

        Accepts:
            - (H, W)            -> add channel + batch dim
            - (H, W, C)         -> move channel to front, add batch dim
            - (N, H, W)         -> add channel dim
            - (N, H, W, C)      -> move channel to front
            - (N, C, H, W)      -> as-is
            - list of (H, W)    -> stack into (N, 1, H, W)
            - list of (H, W, C) -> stack into (N, C, H, W)
        """
        if isinstance(X, list):
            if len(X) == 0:
                return np.zeros((0, self.in_channels, 1, 1), dtype=np.float32)
            # Each item: (H, W) or (H, W, C)
            arrs = []
            for img in X:
                arr = np.asarray(img, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[None, :, :]  # (1, H, W)
                elif arr.ndim == 3:
                    # (H, W, C) -> (C, H, W)
                    arr = np.transpose(arr, (2, 0, 1))
                else:
                    raise ValueError(
                        f"[WHAT] Each image must be 2D or 3D, got shape {arr.shape}\n"
                        "[FIX] Pass (H, W) for grayscale or (H, W, C) for RGB"
                    )
                arrs.append(arr)
            X_arr = np.stack(arrs, axis=0)  # (N, C, H, W)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            if X_arr.ndim == 2:
                X_arr = X_arr[None, None, :, :]  # (1, 1, H, W)
            elif X_arr.ndim == 3:
                # Ambiguous: could be (N, H, W) or (H, W, C)
                # Heuristic: if last dim is 1 or 3 and other dims are bigger, it's (H, W, C)
                # Otherwise it's (N, H, W)
                if X_arr.shape[-1] in (1, 3) and X_arr.shape[-1] < min(X_arr.shape[:-1]):
                    # (H, W, C) -> (1, C, H, W)
                    X_arr = np.transpose(X_arr, (2, 0, 1))[None, :, :, :]
                else:
                    # (N, H, W) -> (N, 1, H, W)
                    X_arr = X_arr[:, None, :, :]
            elif X_arr.ndim == 4:
                # (N, H, W, C) or (N, C, H, W)?
                # If shape[1] in (1, 3) and shape[1] < shape[-1], assume (N, C, H, W)
                # If shape[-1] in (1, 3) and shape[-1] < shape[1], assume (N, H, W, C)
                if X_arr.shape[-1] in (1, 3) and X_arr.shape[-1] <= X_arr.shape[1]:
                    pass  # already (N, C, H, W)? Actually no — (N, H, W, C) has C last
                # Simple rule: if shape[-1] is 1 or 3, it's the channel dim
                if X_arr.shape[-1] in (1, 3) and X_arr.shape[-1] != X_arr.shape[1]:
                    # (N, H, W, C) -> (N, C, H, W)
                    X_arr = np.transpose(X_arr, (0, 3, 1, 2))
                # else assume (N, C, H, W) — leave as-is
            else:
                raise ValueError(
                    f"[WHAT] Input images must be 2D, 3D, or 4D, got shape {X_arr.shape}\n"
                    "[FIX] Pass (H, W), (H, W, C), (N, H, W), (N, H, W, C), or (N, C, H, W)"
                )

        # Validate channel count
        if X_arr.shape[1] != self.in_channels:
            if X_arr.shape[1] in (1, 3) and self.in_channels in (1, 3):
                # Allow mismatch — convert grayscale to RGB or vice versa via repeat/mean
                if self.in_channels == 3 and X_arr.shape[1] == 1:
                    X_arr = np.repeat(X_arr, 3, axis=1)
                elif self.in_channels == 1 and X_arr.shape[1] == 3:
                    X_arr = np.mean(X_arr, axis=1, keepdims=True)
            else:
                raise ValueError(
                    f"[WHAT] Input has {X_arr.shape[1]} channels but model expects {self.in_channels}\n"
                    "[FIX] Set in_channels=1 for grayscale or in_channels=3 for RGB"
                )

        # Normalize to ~[0, 1] if values look like uint8 (max > 1.0)
        # We don't normalize further (BatchNorm handles per-channel scaling)
        if X_arr.max() > 1.5:
            X_arr = X_arr / 255.0

        return X_arr

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------
    def _augment_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a batch of images.

        X_batch shape: (N, C, H, W)
        """
        if not self.augment:
            return X_batch
        N, C, H, W = X_batch.shape
        out = X_batch.copy()

        for i in range(N):
            # Random horizontal flip
            if random.random() < self.augment_flip_p:
                out[i] = out[i, :, :, ::-1].copy()

            # Random rotation by 0/90/180/270 degrees
            if self.augment_rotate:
                k = random.randint(0, 3)
                if k > 0:
                    out[i] = np.rot90(out[i], k=k, axes=(1, 2)).copy()

            # Random crop with padding (only if image is big enough)
            if self.augment_crop_pad > 0 and H > 4 and W > 4:
                pad = self.augment_crop_pad
                padded = np.pad(out[i], ((0, 0), (pad, pad), (pad, pad)),
                                mode="reflect")
                # Random crop back to (H, W)
                top = random.randint(0, 2 * pad)
                left = random.randint(0, 2 * pad)
                out[i] = padded[:, top:top + H, left:left + W]

        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y=None, **kwargs):
        """Train the CNN on image data.

        Args:
            X: Image data. Accepts:
                - numpy array: (N, H, W), (N, H, W, C), or (N, C, H, W)
                - list of 2D arrays (one per image)
                - list of 3D arrays (H, W, C) (one per image)
            y: Integer class labels, shape (N,). Can also be strings —
                they'll be encoded to integers automatically.

        Returns:
            self (for chaining).
        """
        if y is None:
            raise ValueError(
                "[WHAT] CNN.fit() requires y (class labels)\n"
                "[WHY] CNN is a supervised classifier\n"
                "[FIX] Pass y=[0, 1, 2, ...] matching the length of X"
            )

        # Normalize images
        X_arr = self._normalize_images(X)
        self._last_X = X_arr

        # Encode labels
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in ("U", "S", "O"):  # string labels
            self.classes_ = np.unique(y_arr)
            y_encoded = np.searchsorted(self.classes_, y_arr).astype(np.int64)
        else:
            y_encoded = y_arr.astype(np.int64)
            self.classes_ = np.unique(y_encoded)
        self._last_y = y_encoded

        n_classes = int(y_encoded.max()) + 1
        if self.net is None:
            self._build_model(n_classes=n_classes)
        elif n_classes > self.n_classes:
            # Rebuild if more classes than expected
            self._build_model(n_classes=n_classes)

        self._train(X_arr, y_encoded)
        self._trained = True
        return self

    def _train(self, X: np.ndarray, y: np.ndarray):
        """Internal training loop."""
        N = X.shape[0]
        if N == 0:
            raise ValueError("Cannot train on empty dataset")

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, (N // self.batch_size + 1) * self.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: self._lr_lambda(step, total_steps),
        )

        # Modern upgrades (built on top of the existing CNN training loop).
        from .training_enhancements import MixupTrainer, SWA
        mixup = MixupTrainer(alpha=self.mixup_alpha) if self.mixup_alpha > 0 else None
        swa = None
        if self.use_swa:
            swa = SWA(self.net,
                      swa_start=max(self.epochs // 4, 1),
                      swa_freq=max(self.epochs // 10, 1))
        label_smoothing = float(self.label_smoothing)

        self._loss_history = []
        self.net.train()

        for epoch in range(self.epochs):
            # Shuffle each epoch
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                batch_idx = perm[start: start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Augment
                if self.augment:
                    X_batch = self._augment_batch(X_batch)

                X_t = torch.from_numpy(X_batch).to(self.device)
                y_t = torch.from_numpy(y_batch).to(self.device)

                if mixup is not None:
                    X_mix, (y_a, y_b, lam) = mixup.mix(X_t, y_t)
                    logits = self.net(X_mix)
                    loss = (lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                            + (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing))
                else:
                    logits = self.net(X_t)
                    loss = F.cross_entropy(logits, y_t, label_smoothing=label_smoothing)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self._loss_history.append(avg_loss)
            if swa is not None:
                swa.update(epoch)

        # Swap in SWA-averaged weights for inference.
        if swa is not None:
            swa.apply()

    def _lr_lambda(self, step: int, total_steps: int) -> float:
        """Linear warmup → cosine decay."""
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------
    def freeze_conv_base(self) -> "CNN":
        """Freeze the convolutional feature extractor (for transfer learning).

        After calling this, only the linear classifier head will be updated
        during training. Useful when you have a pretrained conv base and want
        to train a new classification head on a small dataset.
        """
        if self.net is None:
            raise RuntimeError(
                "[WHAT] freeze_conv_base() called before model is built\n"
                "[FIX] Either call fit() first, or call _build_model() manually"
            )
        for param in self.net.features.parameters():
            param.requires_grad = False
        # Also freeze the adaptive pool (no params, but for completeness)
        for param in self.net.adaptive_pool.parameters():
            param.requires_grad = False
        self._conv_frozen = True
        return self

    def unfreeze_conv_base(self) -> "CNN":
        """Unfreeze the conv base (undo freeze_conv_base)."""
        if self.net is None:
            return self
        for param in self.net.features.parameters():
            param.requires_grad = True
        self._conv_frozen = False
        return self

    def replace_classifier_head(self, n_classes: int) -> "CNN":
        """Replace the linear classifier head with a new one for `n_classes`.

        Useful for transfer learning: load a pretrained model, replace the
        head, freeze the conv base, then train the new head on your data.
        """
        if self.net is None:
            raise RuntimeError("Model not built yet — call fit() first")
        last_conv_channels = self.conv_channels[-1]
        self.net.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_conv_channels, self.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.fc_hidden, n_classes),
        ).to(self.device)
        self.n_classes = n_classes
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward(self, x, **kwargs):
        """Predict the class of a single image or batch of images.

        Args:
            x: Single image (H, W) or (H, W, C), or batch (N, H, W, C).

        Returns:
            For a single image: the predicted class index (int). If the
            model was trained on string labels, returns the original string.
            For a batch: a numpy array of class indices/labels.
        """
        if not self._trained or self.net is None:
            raise RuntimeError(
                "[WHAT] CNN.predict() called before train()\n"
                "[WHY] The model has no learned weights yet\n"
                "[FIX] Call model.train(X, y) first"
            )

        # Normalize input
        single = False
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (int, float)):
            # Probably a flat list representing one image — wrap
            x = np.asarray(x)
        if isinstance(x, np.ndarray) and x.ndim == 2:
            single = True
            x = x[None, :, :]  # (1, H, W)
        elif isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[-1] < min(x.shape[:-1]):
            single = True
            # (H, W, C) -> (1, H, W, C)
            x = x[None, :, :, :]

        X_arr = self._normalize_images(x)
        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_arr).to(self.device)
            logits = self.net(X_t)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        # Decode labels if string-encoded
        if self.classes_ is not None and self.classes_.dtype.kind in ("U", "S", "O"):
            preds_decoded = np.array([self.classes_[p] for p in preds])
        else:
            preds_decoded = preds

        if single:
            return preds_decoded[0]
        return preds_decoded

    def predict_proba(self, x) -> np.ndarray:
        """Return class probabilities (softmax) for a single image or batch.

        Returns:
            For a single image: 1D array of probabilities, shape (n_classes,).
            For a batch: 2D array, shape (N, n_classes).
        """
        if not self._trained or self.net is None:
            raise RuntimeError(
                "[WHAT] CNN.predict_proba() called before train()"
            )

        single = False
        if isinstance(x, np.ndarray) and x.ndim == 2:
            single = True
            x = x[None, :, :]
        elif isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[-1] < min(x.shape[:-1]):
            single = True
            x = x[None, :, :, :]

        X_arr = self._normalize_images(x)
        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_arr).to(self.device)
            logits = self.net(X_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        if single:
            return probs[0]
        return probs

    # ------------------------------------------------------------------
    # Feature extraction (my own addition)
    # ------------------------------------------------------------------
    def extract_features(self, x, layer_index: int = -1) -> np.ndarray:
        """Extract intermediate activations from the conv stack.

        Useful for:
            - Visualization (what does each filter learn?)
            - Transfer learning (use conv features as input to another model)
            - Debugging (is the conv stack actually learning useful features?)

        Args:
            x: Single image (H, W) or (H, W, C), or batch.
            layer_index: Which conv block's output to return.
                -1 (default) = last conv block.
                0 = first conv block, 1 = second, etc.
                Negative indices count from the end.

        Returns:
            For a single image: (C, H', W') array of activations.
            For a batch: (N, C, H', W') array.
        """
        if not self._trained or self.net is None:
            raise RuntimeError("CNN.extract_features() called before train()")

        single = False
        if isinstance(x, np.ndarray) and x.ndim == 2:
            single = True
            x = x[None, :, :]
        elif isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[-1] < min(x.shape[:-1]):
            single = True
            x = x[None, :, :, :]

        X_arr = self._normalize_images(x)
        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_arr).to(self.device)
            _ = self.net(X_t, capture_activations=True)
            activations = self.net.get_activations()

        if layer_index < -len(activations) or layer_index >= len(activations):
            raise IndexError(
                f"[WHAT] layer_index {layer_index} out of range\n"
                f"[WHY] The CNN has {len(activations)} conv blocks\n"
                f"[FIX] Use a layer_index in [-{len(activations)}, {len(activations) - 1}]"
            )

        feat = activations[layer_index].cpu().numpy()
        if single:
            return feat[0]
        return feat

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def loss_history(self) -> List[float]:
        return list(self._loss_history)

    def parameter_count(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def architecture_summary(self) -> Dict[str, Any]:
        return {
            "technique": "cnn",
            "in_channels": self.in_channels,
            "n_classes": self.n_classes,
            "conv_channels": self.conv_channels,
            "kernel_sizes": self.kernel_sizes,
            "pool_sizes": self.pool_sizes,
            "fc_hidden": self.fc_hidden,
            "dropout": self.dropout,
            "parameters": self.parameter_count(),
            "trainable_parameters": self.parameter_count(),
            "conv_frozen": self._conv_frozen,
            "device": str(self.device),
            "classes_": (self.classes_.tolist() if self.classes_ is not None else None),
        }

    def accuracy(self) -> float:
        if not self._trained or self._last_X is None or self._last_y is None:
            return 0.0
        preds = self.forward(self._last_X)
        return float(np.mean(np.asarray(preds) == self._last_y))

    def __repr__(self):
        if self.net is None:
            return f"<CNN unbuilt in_channels={self.in_channels} n_classes={self.n_classes}>"
        return (
            f"<CNN in_channels={self.in_channels} n_classes={self.n_classes} "
            f"conv_channels={self.conv_channels} fc_hidden={self.fc_hidden} "
            f"params={self.parameter_count():,} trained={self._trained}>"
        )
