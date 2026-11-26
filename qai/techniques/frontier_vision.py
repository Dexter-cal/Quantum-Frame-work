"""
qai.techniques.frontier_vision -- vision-specific architectures.

  qai.build(technique="vit", ...)         — Vision Transformer (Dosovitskiy 2021)
  qai.build(technique="deit", ...)        — Data-efficient Image Transformer
  qai.build(technique="nerf", ...)        — Neural Radiance Field (Mildenhall 2020)
  qai.build(technique="inr", ...)         — Implicit Neural Representation
  qai.build(technique="unet", ...)        — U-Net for image segmentation
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError(
            "[WHAT] Frontier vision techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# 1. VISION TRANSFORMER (Dosovitskiy et al. 2021)
# =====================================================================
class _PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_ch=1, emb_dim=64):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class _ViTBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_dim=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(Technique):
    """ViT — image classification via transformer on patch embeddings.

    Image is split into patches, each patch is linearly embedded, a [CLS]
    token is prepended, positional embeddings are added, and the sequence
    goes through L transformer blocks. Final [CLS] embedding is used for
    classification.
    """
    name = "vit"
    family = "vision"

    def __init__(self, img_size: int = 28, patch_size: int = 4,
                 in_ch: int = 1, n_classes: int = 10, emb_dim: int = 64,
                 depth: int = 4, n_heads: int = 4, mlp_dim: int = 128,
                 lr: float = 1e-3, epochs: int = 30, **kw):
        super().__init__(img_size=img_size, patch_size=patch_size, in_ch=in_ch,
                         n_classes=n_classes, emb_dim=emb_dim, depth=depth,
                         n_heads=n_heads, mlp_dim=mlp_dim, lr=lr, epochs=epochs, **kw)
        _req()
        self.img_size, self.patch_size = img_size, patch_size
        self.n_classes, self.emb_dim = n_classes, emb_dim
        self.lr, self.epochs = lr, epochs
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_ch, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        n_patches = self.patch_embed.n_patches
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([_ViTBlock(emb_dim, n_heads, mlp_dim) for _ in range(depth)])
        self.ln = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        params = [self.cls_token, self.pos_emb]
        params += list(self.patch_embed.parameters())
        for b in self.blocks:
            params += list(b.parameters())
        params += list(self.ln.parameters())
        params += list(self.head.parameters())
        return params

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if X.ndim == 3:
            X = X[:, None]  # add channel dim
        N = X.shape[0]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            x = torch.from_numpy(X)
            x = self.patch_embed(x)
            cls = self.cls_token.expand(N, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_emb
            for blk in self.blocks:
                x = blk(x)
            x = self.ln(x[:, 0])
            logits = self.head(x)
            loss = F.cross_entropy(logits, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            e = self.patch_embed(x)
            cls = self.cls_token.expand(1, -1, -1)
            h = torch.cat([cls, e], dim=1) + self.pos_emb
            for blk in self.blocks:
                h = blk(h)
            h = self.ln(h[:, 0])
            return self.head(h).argmax(-1).item()


# =====================================================================
# 2. DEIT — Data-efficient Image Transformer (Touvron 2021)
# =====================================================================
class DeiT(VisionTransformer):
    """DeiT — ViT with a distillation token and hard-label teacher distillation.

    Same as ViT but adds a [DIST] token trained to mimic a teacher's
    predictions in addition to the [CLS] token's classification loss.
    """
    name = "deit"
    family = "vision"

    def __init__(self, *args, **kw):
        # Initialize dist_token BEFORE super().__init__() so that the
        # overridden parameters() (which references self.dist_token) is valid
        # when the parent's optimizer is constructed.
        # We don't know emb_dim yet, so we initialize dist_token lazily and
        # use a guard in parameters().
        self.dist_token = None
        self.head_dist = None
        super().__init__(*args, **kw)
        # Now emb_dim is known — create the real dist_token + head_dist
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        self.head_dist = nn.Linear(self.emb_dim, self.n_classes)
        # Re-create the optimizer so it includes the new params
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Override pos_emb to account for [CLS] + [DIST] tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 2, self.emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)

    def parameters(self):
        params = super().parameters()
        if self.dist_token is not None:
            params = [self.dist_token] + list(params)
        if self.head_dist is not None:
            params = list(params) + list(self.head_dist.parameters())
        return params

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if X.ndim == 3:
            X = X[:, None]
        N = X.shape[0]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            x = torch.from_numpy(X)
            x = self.patch_embed(x)
            cls = self.cls_token.expand(N, -1, -1)
            dist = self.dist_token.expand(N, -1, -1)
            x = torch.cat([cls, dist, x], dim=1)
            x = x + self.pos_emb
            for blk in self.blocks:
                x = blk(x)
            x = self.ln(x)
            cls_out = self.head(x[:, 0])
            dist_out = self.head_dist(x[:, 1])
            # combine logits; here we use simple average since no teacher given
            logits = 0.5 * cls_out + 0.5 * dist_out
            loss = F.cross_entropy(logits, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            e = self.patch_embed(x)
            cls = self.cls_token.expand(1, -1, -1)
            dist = self.dist_token.expand(1, -1, -1)
            h = torch.cat([cls, dist, e], dim=1) + self.pos_emb
            for blk in self.blocks:
                h = blk(h)
            h = self.ln(h)
            logits = 0.5 * self.head(h[:, 0]) + 0.5 * self.head_dist(h[:, 1])
            return logits.argmax(-1).item()


# =====================================================================
# 3. NeRF — Neural Radiance Field (Mildenhall et al. 2020)
# =====================================================================
class _NeRFNet(nn.Module):
    def __init__(self, pos_dim=6, dir_dim=3, hidden=64):
        super().__init__()
        self.pos_dim = pos_dim
        self.dir_dim = dir_dim
        # positional encoding via Fourier features
        self.feat = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # density + color head
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, pos, dir):
        h = self.feat(pos)
        sigma = F.softplus(self.sigma(h))
        c = self.color(torch.cat([h, dir], dim=-1))
        return sigma, c


class NeRF(Technique):
    """NeRF — learn a 5D radiance field from posed images.

    A small MLP maps (position, direction) -> (density, RGB color). Volume
    rendering integrates along rays to produce pixels. Training minimises
    MSE between rendered and ground-truth pixels.
    """
    name = "nerf"
    family = "vision"

    def __init__(self, pos_dim: int = 6, dir_dim: int = 3, hidden: int = 64,
                 n_samples: int = 32, lr: float = 5e-4,
                 epochs: int = 100, **kw):
        super().__init__(pos_dim=pos_dim, dir_dim=dir_dim, hidden=hidden,
                         n_samples=n_samples, lr=lr, epochs=epochs, **kw)
        _req()
        self.pos_dim, self.dir_dim = pos_dim, dir_dim
        self.n_samples = n_samples
        self.lr, self.epochs = lr, epochs
        self.net = _NeRFNet(pos_dim, dir_dim, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _render_rays(self, origins, dirs, near=0.0, far=1.0):
        # origins: (N, 3), dirs: (N, 3)
        N = origins.shape[0]
        z = torch.linspace(near, far, self.n_samples)
        pts = origins.unsqueeze(1) + dirs.unsqueeze(1) * z.view(1, -1, 1)
        pts = pts.view(N * self.n_samples, 3)
        dirs_exp = dirs.unsqueeze(1).expand(-1, self.n_samples, -1).reshape(-1, 3)
        # encode positions with simple Fourier features (use raw 3D here)
        sigma, color = self.net(pts, dirs_exp)
        sigma = sigma.view(N, self.n_samples)
        color = color.view(N, self.n_samples, 3)
        # volume rendering
        delta = torch.full_like(sigma, (far - near) / self.n_samples)
        alpha = 1 - torch.exp(-sigma * delta)
        T = torch.cumprod(1 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones(N, 1), T[:, :-1]], dim=-1)
        w = T * alpha
        rgb = (w.unsqueeze(-1) * color).sum(1)
        return rgb

    def fit(self, X, y):
        """X: dict with 'origins' (N,3), 'dirs' (N,3); y: (N,3) target colors."""
        origins = torch.as_tensor(np.asarray(X["origins"], dtype=np.float32))
        dirs = torch.as_tensor(np.asarray(X["dirs"], dtype=np.float32))
        colors = torch.as_tensor(np.asarray(y, dtype=np.float32))
        for ep in range(self.epochs):
            self.opt.zero_grad()
            rgb = self._render_rays(origins, dirs)
            loss = F.mse_loss(rgb, colors)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def render(self, origins, dirs):
        origins = torch.as_tensor(np.asarray(origins, dtype=np.float32))
        dirs = torch.as_tensor(np.asarray(dirs, dtype=np.float32))
        with torch.no_grad():
            return self._render_rays(origins, dirs).numpy()

    def forward(self, x):
        return self.render(x["origins"], x["dirs"])


# =====================================================================
# 4. INR — Implicit Neural Representation
# =====================================================================
class INR(Technique):
    """INR — coordinate-based MLP that maps (x, y) -> RGB.

    Used to compress images (and 3D shapes) by representing the signal as
    the weights of a small network rather than a discrete grid. Often uses
    sinusoidal activations (SIREN) for high-frequency detail.
    """
    name = "inr"
    family = "vision"

    def __init__(self, coord_dim: int = 2, out_dim: int = 3, hidden: int = 64,
                 depth: int = 4, omega: float = 30.0,
                 lr: float = 5e-4, epochs: int = 100, **kw):
        super().__init__(coord_dim=coord_dim, out_dim=out_dim, hidden=hidden,
                         depth=depth, omega=omega, lr=lr, epochs=epochs, **kw)
        _req()
        self.coord_dim, self.out_dim = coord_dim, out_dim
        self.omega = omega
        self.lr, self.epochs = lr, epochs
        layers = [nn.Linear(coord_dim, hidden)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.ModuleList(layers)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        return list(self.net.parameters())

    def _siren_forward(self, x):
        x = self.omega * x
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:
                x = torch.sin(self.omega * x)
        return x

    def fit(self, X, y):
        """X: (N, coord_dim) coordinates; y: (N, out_dim) target values."""
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = torch.as_tensor(np.asarray(y, dtype=np.float32))
        for ep in range(self.epochs):
            self.opt.zero_grad()
            out = self._siren_forward(X)
            loss = F.mse_loss(out, y)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self._siren_forward(x).squeeze(0).numpy()


# =====================================================================
# 5. U-NET — image segmentation
# =====================================================================
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        dy = x2.shape[2] - x1.shape[2]
        dx = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(Technique):
    """U-Net — encoder-decoder with skip connections for segmentation.

    The encoder progressively downsamples; the decoder upsamples and
    concatenates features from the matching encoder layer. Standard for
    biomedical image segmentation and many diffusion models' U-Net.
    """
    name = "unet"
    family = "vision"

    def __init__(self, in_ch: int = 1, n_classes: int = 2,
                 base: int = 16, img_size: int = 32,
                 lr: float = 1e-3, epochs: int = 30, **kw):
        super().__init__(in_ch=in_ch, n_classes=n_classes, base=base,
                         img_size=img_size, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_ch, self.n_classes = in_ch, n_classes
        self.lr, self.epochs = lr, epochs
        b = base
        self.inc = _DoubleConv(in_ch, b)
        self.d1 = _Down(b, b * 2)
        self.d2 = _Down(b * 2, b * 4)
        self.d3 = _Down(b * 4, b * 8)
        self.u3 = _Up(b * 8, b * 4)
        self.u2 = _Up(b * 4, b * 2)
        self.u1 = _Up(b * 2, b)
        self.outc = nn.Conv2d(b, n_classes, 1)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        params = []
        for m in [self.inc, self.d1, self.d2, self.d3, self.u3, self.u2, self.u1, self.outc]:
            params += list(m.parameters())
        return params

    def _forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x = self.u3(x4, x3)
        x = self.u2(x, x2)
        x = self.u1(x, x1)
        return self.outc(x)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if X.ndim == 3:
            X = X[:, None]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            x = torch.from_numpy(X)
            seg = self._forward(x)
            loss = F.cross_entropy(seg, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            seg = self._forward(x)
            return seg.argmax(1).squeeze(0).numpy()
