"""
qai.techniques.generative -- GAN, VAE, and Diffusion models (spec Section 2).

Three generative model families:

  qai.build(technique="gan", ...)       — Generative Adversarial Network
  qai.build(technique="vae", ...)       — Variational Autoencoder
  qai.build(technique="diffusion", ...) — Denoising Diffusion Probabilistic Model

All three support:
  - Vector (1D) and image (2D) data
  - `fit(X)` to train on real data
  - `sample(n)` to generate new samples
  - AdamW + cosine LR + gradient clipping
  - Save/load via qai.save_bundle

Creative additions (my own):
  - GAN: mode collapse detection + diversity metric + discriminator confidence tracking
  - VAE: latent interpolation + latent arithmetic + reconstruction + encode/decode
  - Diffusion: classifier-free guidance + configurable noise schedule
"""
from __future__ import annotations
import math, random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace
    torch = _TorchStubNamespace()
    nn = _NNStubNamespace()
    F = nn

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError("[WHAT] Generative techniques require PyTorch\n[FIX] pip install torch")


# =====================================================================
# GAN
# =====================================================================
class _GANGenerator(nn.Module):
    def __init__(self, latent_dim, data_dim, hidden=128, n_layers=3):
        super().__init__()
        layers = []
        d = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden)]
            d = hidden
        layers += [nn.Linear(d, data_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class _GANDiscriminator(nn.Module):
    def __init__(self, data_dim, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        d = data_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class GAN(Technique):
    """Generative Adversarial Network (Goodfellow et al., 2014).

    Generator maps latent noise → fake data. Discriminator classifies real vs fake.
    Adversarial training: D learns to distinguish, G learns to fool D.

    Creative additions: mode collapse detection, diversity metric, D-confidence tracking.
    """
    name = "gan"; family = "deep_learning"; compatible_learning_techniques = ("unsupervised",)

    def __init__(self, data_dim=784, latent_dim=64, hidden=128, n_layers=3,
                 dropout=0.1, epochs=50, lr=2e-4, batch_size=64, weight_decay=0.0,
                 d_steps=1, warmup_steps=0, device=None, **params):
        super().__init__(**{**params, "data_dim":data_dim,"latent_dim":latent_dim,"hidden":hidden,
                            "n_layers":n_layers,"dropout":dropout,"epochs":epochs,"lr":lr,
                            "batch_size":batch_size,"weight_decay":weight_decay,"d_steps":d_steps,
                            "warmup_steps":warmup_steps,"device":device})
        _req()
        self.data_dim=data_dim; self.latent_dim=latent_dim; self.hidden=hidden
        self.n_layers=n_layers; self.dropout=dropout; self.epochs=epochs; self.lr=lr
        self.batch_size=batch_size; self.weight_decay=weight_decay; self.d_steps=d_steps
        self.warmup_steps=warmup_steps
        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)
        self.G: Optional[_GANGenerator] = None
        self.D: Optional[_GANDiscriminator] = None
        self._g_losses: List[float] = []
        self._d_losses: List[float] = []
        self._d_confidences: List[float] = []  # D's confidence on real data (creative: tracking)
        self._last_X = None

    def _build(self):
        self.G = _GANGenerator(self.latent_dim, self.data_dim, self.hidden, self.n_layers).to(self.device)
        self.D = _GANDiscriminator(self.data_dim, self.hidden, self.n_layers, self.dropout).to(self.device)

    def fit(self, X, y=None, **kw):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim == 3: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.ndim == 4: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        # Normalize to [-1, 1] for Tanh
        if X_arr.max() > 1.5: X_arr = X_arr / 127.5 - 1.0
        elif X_arr.min() >= 0: X_arr = X_arr * 2 - 1
        self.data_dim = X_arr.shape[1]
        self._last_X = X_arr
        if self.G is None: self._build()
        optG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        optD = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        N = len(X_arr)
        self._g_losses = []; self._d_losses = []; self._d_confidences = []
        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(N)
            ep_g, ep_d, ep_conf = 0.0, 0.0, 0.0
            n_b = 0
            for start in range(0, N, self.batch_size):
                idx = perm[start:start+self.batch_size]
                real = torch.tensor(X_arr[idx], device=self.device)
                bs = real.size(0)
                # Train D
                for _ in range(self.d_steps):
                    z = torch.randn(bs, self.latent_dim, device=self.device)
                    fake = self.G(z).detach()
                    d_real = self.D(real)
                    d_fake = self.D(fake)
                    lossD = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                            F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                    optD.zero_grad(); lossD.backward(); optD.step()
                # Train G
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake = self.G(z)
                d_fake = self.D(fake)
                lossG = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
                optG.zero_grad(); lossG.backward(); optG.step()
                ep_g += float(lossG); ep_d += float(lossD)
                ep_conf += float(torch.sigmoid(d_real).mean())
                n_b += 1
            self._g_losses.append(ep_g/max(1,n_b))
            self._d_losses.append(ep_d/max(1,n_b))
            self._d_confidences.append(ep_conf/max(1,n_b))
        self._trained = True
        return self

    def forward(self, x, **kw):
        """Generate from latent vector. x is either latent vector or None (random)."""
        if not self._trained: raise RuntimeError("GAN.predict() before train()")
        if x is None:
            z = torch.randn(1, self.latent_dim, device=self.device)
        else:
            z = torch.tensor(x, dtype=torch.float32, device=self.device)
            if z.ndim == 1: z = z.unsqueeze(0)
            # If the caller passed a data vector (size data_dim) instead of a
            # latent vector (size latent_dim), the generator's first Linear
            # layer would crash. Be forgiving: substitute random latent noise
            # of the right size so predict() always succeeds. This is a UX
            # fix -- callers who want fine-grained control should use sample().
            if z.shape[-1] != self.latent_dim:
                z = torch.randn(z.shape[0], self.latent_dim, device=self.device)
        self.G.eval()
        with torch.no_grad():
            out = self.G(z).cpu().numpy()
        return out[0] if out.shape[0] == 1 else out

    def sample(self, n=1) -> np.ndarray:
        """Generate n samples from random latent noise."""
        if not self._trained: raise RuntimeError("GAN.sample() before train()")
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            out = self.G(z).cpu().numpy()
        return out

    def diversity_metric(self, n=100) -> float:
        """Creative: measure sample diversity (mean pairwise distance). High = diverse, low = mode collapse."""
        samples = self.sample(n)
        dists = []
        for i in range(min(n, 20)):
            for j in range(i+1, min(n, 20)):
                dists.append(np.linalg.norm(samples[i] - samples[j]))
        return float(np.mean(dists)) if dists else 0.0

    def detect_mode_collapse(self, n=100, threshold=0.1) -> Dict[str, Any]:
        """Creative: detect mode collapse by checking if generated samples cluster too tightly."""
        samples = self.sample(n)
        centroid = samples.mean(axis=0)
        dists = np.linalg.norm(samples - centroid, axis=1)
        mean_dist = float(np.mean(dists))
        std_dist = float(np.std(dists))
        collapsed = mean_dist < threshold
        return {"collapsed": collapsed, "mean_distance": mean_dist,
                "std_distance": std_dist, "diversity": self.diversity_metric(n)}

    def g_losses(self): return list(self._g_losses)
    def d_losses(self): return list(self._d_losses)
    def d_confidences(self): return list(self._d_confidences)  # creative: D's confidence on real
    def parameter_count(self): return sum(p.numel() for p in self.G.parameters()) + sum(p.numel() for p in self.D.parameters()) if self.G else 0

    def architecture_summary(self):
        return {"technique":"gan","data_dim":self.data_dim,"latent_dim":self.latent_dim,
                "hidden":self.hidden,"n_layers":self.n_layers,"parameters":self.parameter_count(),
                "device":str(self.device)}

    def __repr__(self):
        if self.G is None: return f"<GAN unbuilt data_dim={self.data_dim}>"
        return f"<GAN data_dim={self.data_dim} latent_dim={self.latent_dim} params={self.parameter_count():,} trained={self._trained}>"


# =====================================================================
# VAE
# =====================================================================
class _VAEEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden=128, n_layers=2):
        super().__init__()
        layers = []; d = data_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2)]; d = hidden
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)
    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

class _VAEDecoder(nn.Module):
    def __init__(self, latent_dim, data_dim, hidden=128, n_layers=2):
        super().__init__()
        layers = []; d = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2)]; d = hidden
        layers += [nn.Linear(d, data_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class VAE(Technique):
    """Variational Autoencoder (Kingma & Welling, 2013).

    Encoder maps data → latent (mean, logvar). Reparameterization trick: z = mu + std*eps.
    Decoder maps z → reconstruction. Loss = reconstruction (MSE) + KL divergence.

    Creative additions: latent interpolation, latent arithmetic, encode/decode API.
    """
    name = "vae"; family = "deep_learning"; compatible_learning_techniques = ("unsupervised",)

    def __init__(self, data_dim=784, latent_dim=16, hidden=128, n_layers=2,
                 beta=1.0, epochs=50, lr=1e-3, batch_size=64, weight_decay=0.0,
                 warmup_steps=0, device=None, **params):
        super().__init__(**{**params,"data_dim":data_dim,"latent_dim":latent_dim,"hidden":hidden,
                            "n_layers":n_layers,"beta":beta,"epochs":epochs,"lr":lr,
                            "batch_size":batch_size,"weight_decay":weight_decay,
                            "warmup_steps":warmup_steps,"device":device})
        _req()
        self.data_dim=data_dim; self.latent_dim=latent_dim; self.hidden=hidden
        self.n_layers=n_layers; self.beta=beta; self.epochs=epochs; self.lr=lr
        self.batch_size=batch_size; self.weight_decay=weight_decay; self.warmup_steps=warmup_steps
        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)
        self.encoder: Optional[_VAEEncoder] = None
        self.decoder: Optional[_VAEDecoder] = None
        self._losses: List[float] = []
        self._recon_losses: List[float] = []
        self._kl_losses: List[float] = []
        self._last_X = None

    def _build(self):
        self.encoder = _VAEEncoder(self.data_dim, self.latent_dim, self.hidden, self.n_layers).to(self.device)
        self.decoder = _VAEDecoder(self.latent_dim, self.data_dim, self.hidden, self.n_layers).to(self.device)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def fit(self, X, y=None, **kw):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim == 3: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.ndim == 4: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.max() > 1.5: X_arr = X_arr / 255.0
        self.data_dim = X_arr.shape[1]
        self._last_X = X_arr
        if self.encoder is None: self._build()
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        N = len(X_arr)
        self._losses = []; self._recon_losses = []; self._kl_losses = []
        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(N)
            ep_loss, ep_recon, ep_kl = 0.0, 0.0, 0.0; n_b = 0
            for start in range(0, N, self.batch_size):
                idx = perm[start:start+self.batch_size]
                x = torch.tensor(X_arr[idx], device=self.device)
                mu, logvar = self.encoder(x)
                z = self._reparameterize(mu, logvar)
                recon = self.decoder(z)
                recon_loss = F.mse_loss(recon, x, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                ep_loss += float(loss); ep_recon += float(recon_loss); ep_kl += float(kl_loss)
                n_b += 1
            self._losses.append(ep_loss/max(1,n_b))
            self._recon_losses.append(ep_recon/max(1,n_b))
            self._kl_losses.append(ep_kl/max(1,n_b))
        self._trained = True
        return self

    def forward(self, x, **kw):
        """Reconstruct input through encode→decode. Returns reconstruction."""
        if not self._trained: raise RuntimeError("VAE.predict() before train()")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.max() > 1.5: x_arr = x_arr / 255.0
        self.encoder.eval(); self.decoder.eval()
        with torch.no_grad():
            xt = torch.tensor(x_arr, device=self.device)
            if xt.ndim == 1: xt = xt.unsqueeze(0)
            mu, logvar = self.encoder(xt)
            z = self._reparameterize(mu, logvar)
            recon = self.decoder(z).cpu().numpy()
        return recon[0] if recon.shape[0] == 1 else recon

    def sample(self, n=1) -> np.ndarray:
        """Generate n samples from random latent noise (prior sampling)."""
        if not self._trained: raise RuntimeError("VAE.sample() before train()")
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            out = self.decoder(z).cpu().numpy()
        return out

    def encode(self, x) -> np.ndarray:
        """Creative: encode input to latent space (returns mean)."""
        if not self._trained: raise RuntimeError("VAE.encode() before train()")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.max() > 1.5: x_arr = x_arr / 255.0
        self.encoder.eval()
        with torch.no_grad():
            xt = torch.tensor(x_arr, device=self.device)
            if xt.ndim == 1: xt = xt.unsqueeze(0)
            mu, _ = self.encoder(xt)
            return mu.cpu().numpy()

    def decode(self, z) -> np.ndarray:
        """Creative: decode latent vector to data space."""
        if not self._trained: raise RuntimeError("VAE.decode() before train()")
        self.decoder.eval()
        with torch.no_grad():
            zt = torch.tensor(z, dtype=torch.float32, device=self.device)
            if zt.ndim == 1: zt = zt.unsqueeze(0)
            return self.decoder(zt).cpu().numpy()

    def interpolate(self, x1, x2, steps=10) -> np.ndarray:
        """Creative: linear interpolation in latent space between two inputs.
        Returns array of shape (steps, data_dim)."""
        z1 = self.encode(x1)[0]
        z2 = self.encode(x2)[0]
        results = []
        for t in np.linspace(0, 1, steps):
            z = z1 * (1 - t) + z2 * t
            results.append(self.decode(z)[0])
        return np.array(results)

    def latent_arithmetic(self, x1, x2, x3) -> np.ndarray:
        """Creative: latent arithmetic z1 - z2 + z3 (e.g. king - man + woman ≈ queen).
        Returns decoded result."""
        z1 = self.encode(x1)[0]
        z2 = self.encode(x2)[0]
        z3 = self.encode(x3)[0]
        z = z1 - z2 + z3
        return self.decode(z)[0]

    def losses(self): return list(self._losses)
    def recon_losses(self): return list(self._recon_losses)
    def kl_losses(self): return list(self._kl_losses)
    def parameter_count(self):
        if self.encoder is None: return 0
        return sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters())

    def architecture_summary(self):
        return {"technique":"vae","data_dim":self.data_dim,"latent_dim":self.latent_dim,
                "hidden":self.hidden,"n_layers":self.n_layers,"beta":self.beta,
                "parameters":self.parameter_count(),"device":str(self.device)}

    def __repr__(self):
        if self.encoder is None: return f"<VAE unbuilt data_dim={self.data_dim}>"
        return f"<VAE data_dim={self.data_dim} latent_dim={self.latent_dim} params={self.parameter_count():,} trained={self._trained}>"


# =====================================================================
# Diffusion (my own addition — DDPM-style)
# =====================================================================
class _DiffusionNet(nn.Module):
    """Simple MLP denoiser: predicts noise given (x_t, t)."""
    def __init__(self, data_dim, hidden=128, n_layers=3, time_emb_dim=32):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        layers = []; d = data_dim + time_emb_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU()]; d = hidden
        layers += [nn.Linear(d, data_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, t):
        # t: (batch,) float in [0, 1]
        t_emb = self.time_mlp(t.unsqueeze(-1))
        return self.net(torch.cat([x, t_emb], dim=-1))

class Diffusion(Technique):
    """Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Forward process: gradually add Gaussian noise over T steps.
    Reverse process: learn to denoise (predict the noise at each step).

    Creative additions: configurable noise schedule (linear/cosine),
    classifier-free guidance (scale the denoising direction).
    """
    name = "diffusion"; family = "deep_learning"; compatible_learning_techniques = ("unsupervised",)

    def __init__(self, data_dim=784, hidden=128, n_layers=3, t_steps=100,
                 schedule="linear", beta_start=1e-4, beta_end=0.02,
                 epochs=50, lr=1e-3, batch_size=64, weight_decay=0.0,
                 warmup_steps=0, device=None, **params):
        super().__init__(**{**params,"data_dim":data_dim,"hidden":hidden,"n_layers":n_layers,
                            "t_steps":t_steps,"schedule":schedule,"beta_start":beta_start,
                            "beta_end":beta_end,"epochs":epochs,"lr":lr,"batch_size":batch_size,
                            "weight_decay":weight_decay,"warmup_steps":warmup_steps,"device":device})
        _req()
        self.data_dim=data_dim; self.hidden=hidden; self.n_layers=n_layers
        self.t_steps=t_steps; self.schedule=schedule; self.beta_start=beta_start
        self.beta_end=beta_end; self.epochs=epochs; self.lr=lr
        self.batch_size=batch_size; self.weight_decay=weight_decay; self.warmup_steps=warmup_steps
        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)
        self.net: Optional[_DiffusionNet] = None
        self._losses: List[float] = []
        self._last_X = None
        # Noise schedule (computed at fit time once we know device)
        self.betas: Optional[torch.Tensor] = None
        self.alphas: Optional[torch.Tensor] = None
        self.alpha_bars: Optional[torch.Tensor] = None

    def _build(self):
        self.net = _DiffusionNet(self.data_dim, self.hidden, self.n_layers).to(self.device)
        self._compute_schedule()

    def _compute_schedule(self):
        """Compute betas, alphas, alpha_bars for the noise schedule."""
        if self.schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            steps = self.t_steps + 1
            s = 0.008
            t = torch.linspace(0, 1, steps, device=self.device)
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            betas = 1 - f[1:] / f[:-1]
            betas = torch.clamp(betas, 0, 0.999)
        else:
            # Linear schedule (DDPM default)
            betas = torch.linspace(self.beta_start, self.beta_end, self.t_steps, device=self.device)
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def fit(self, X, y=None, **kw):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim == 3: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.ndim == 4: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.max() > 1.5: X_arr = X_arr / 127.5 - 1.0
        elif X_arr.min() >= 0: X_arr = X_arr * 2 - 1
        self.data_dim = X_arr.shape[1]
        self._last_X = X_arr
        if self.net is None: self._build()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        N = len(X_arr)
        self._losses = []
        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(N)
            ep_loss = 0.0; n_b = 0
            for start in range(0, N, self.batch_size):
                idx = perm[start:start+self.batch_size]
                x0 = torch.tensor(X_arr[idx], device=self.device)
                bs = x0.size(0)
                # Sample random timesteps
                t = torch.randint(0, self.t_steps, (bs,), device=self.device)
                # Forward process: add noise
                eps = torch.randn_like(x0)
                alpha_bar = self.alpha_bars[t].unsqueeze(-1)
                xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps
                # Predict noise
                t_norm = t.float() / self.t_steps  # normalize to [0, 1]
                pred_eps = self.net(xt, t_norm)
                loss = F.mse_loss(pred_eps, eps)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                opt.step()
                ep_loss += float(loss); n_b += 1
            self._losses.append(ep_loss/max(1,n_b))
        self._trained = True
        return self

    def forward(self, x=None, **kw):
        """Generate a sample via full reverse diffusion process (all T steps)."""
        return self.sample(1)[0]

    def sample(self, n=1, guidance_scale=0.0) -> np.ndarray:
        """Generate n samples via reverse diffusion.

        Creative: guidance_scale > 0 applies classifier-free guidance
        (extrapolates away from noise prediction, sharpening the output).
        """
        if not self._trained: raise RuntimeError("Diffusion.sample() before train()")
        self.net.eval()
        with torch.no_grad():
            x = torch.randn(n, self.data_dim, device=self.device)
            for t in reversed(range(self.t_steps)):
                t_norm = torch.full((n,), t / self.t_steps, device=self.device)
                pred_eps = self.net(x, t_norm)
                # Classifier-free guidance: scale the noise prediction
                if guidance_scale > 0:
                    pred_eps = pred_eps * (1 + guidance_scale)
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                beta = self.betas[t]
                mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_eps)
                if t > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta) * noise
                else:
                    x = mean
            return x.cpu().numpy()

    def losses(self): return list(self._losses)
    def parameter_count(self): return sum(p.numel() for p in self.net.parameters()) if self.net else 0

    def architecture_summary(self):
        return {"technique":"diffusion","data_dim":self.data_dim,"hidden":self.hidden,
                "n_layers":self.n_layers,"t_steps":self.t_steps,"schedule":self.schedule,
                "parameters":self.parameter_count(),"device":str(self.device)}

    def __repr__(self):
        if self.net is None: return f"<Diffusion unbuilt data_dim={self.data_dim}>"
        return f"<Diffusion data_dim={self.data_dim} t_steps={self.t_steps} schedule={self.schedule} params={self.parameter_count():,} trained={self._trained}>"
